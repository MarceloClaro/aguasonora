import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score
import librosa
import librosa.display
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from scipy.io import wavfile
import tempfile
import shutil
import requests

# Suprimir avisos de classes Torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.classes.*")

# Configurações de gráficos
sns.set_style('whitegrid')

# Configuração de dispositivo (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definir seed para reprodutibilidade
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Chamar a configuração de seed
# Função para baixar o SoundFont
def download_soundfont():
    """
    Verifica se o SoundFont FluidR3_GM.sf2 existe no diretório 'sounds'.
    Caso não exista, realiza o download automaticamente.
    """
    soundfont_dir = 'sounds'
    soundfont_filename = 'FluidR3_GM.sf2'
    soundfont_path = os.path.join(soundfont_dir, soundfont_filename)

    if not os.path.exists(soundfont_path):
        st.info("SoundFont necessário não encontrado. Baixando...")
        os.makedirs(soundfont_dir, exist_ok=True)
        try:
            sf_url = "https://musical-artifacts.com/artifacts/738/FluidR3_GM.sf2/download"
            response = requests.get(sf_url, stream=True)
            response.raise_for_status()
            with open(soundfont_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            st.success("SoundFont baixado com sucesso.")
        except Exception as e:
            st.error(f"Erro ao baixar o SoundFont: {e}")
            return None
    else:
        st.success("SoundFont encontrado e configurado.")
    
    return soundfont_path

# Configuração inicial para SoundFont
soundfont_path = download_soundfont()
if soundfont_path is None:
    st.stop()  # Interrompe a execução se o SoundFont não for encontrado ou baixado.
# Função para garantir taxa de amostragem e formato mono
def ensure_sample_rate(original_sr, waveform, desired_sr=16000):
    """
    Ajusta a taxa de amostragem para o áudio, se necessário, para garantir 16 kHz.
    """
    if original_sr != desired_sr:
        desired_length = int(round(float(len(waveform)) / original_sr * desired_sr))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sr, waveform

# Upload e exibição do áudio
def upload_and_process_audio():
    """
    Permite o upload de um arquivo de áudio e realiza o pré-processamento.
    """
    st.header("Upload de Áudio")
    st.write("Carregue um arquivo de áudio para análise e processamento.")

    uploaded_audio = st.file_uploader(
        "Faça upload do arquivo de áudio (formatos suportados: WAV, MP3, OGG, FLAC)", 
        type=["wav", "mp3", "ogg", "flac"]
    )

    if uploaded_audio is not None:
        # Salvar arquivo temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmp_audio:
            tmp_audio.write(uploaded_audio.read())
            tmp_audio_path = tmp_audio.name

        # Processar e exibir informações do áudio
        try:
            # Ler o áudio
            sample_rate, wav_data = wavfile.read(tmp_audio_path)
            
            # Verificar se o áudio é estéreo e convertê-lo para mono
            if wav_data.ndim > 1:
                wav_data = wav_data.mean(axis=1)

            # Garantir taxa de amostragem correta
            if sample_rate != 16000:
                sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

            # Normalizar áudio para [-1, 1]
            if wav_data.dtype.kind == 'i':
                wav_data = wav_data / np.iinfo(wav_data.dtype).max
            elif wav_data.dtype.kind == 'f':
                if np.max(wav_data) > 1.0 or np.min(wav_data) < -1.0:
                    wav_data = wav_data / np.max(np.abs(wav_data))
            wav_data = wav_data.astype(np.float32)

            # Exibir informações do áudio
            st.write(f"**Sample rate:** {sample_rate} Hz")
            st.write(f"**Duração total:** {len(wav_data) / sample_rate:.2f}s")
            st.write(f"**Tamanho dos dados:** {len(wav_data)} amostras")
            st.audio(wav_data, format='audio/wav', sample_rate=sample_rate)

            return wav_data, sample_rate, tmp_audio_path

        except Exception as e:
            st.error(f"Erro ao processar o áudio: {e}")
            return None, None, None
    else:
        st.warning("Nenhum arquivo foi carregado.")
        return None, None, None

# Chamada da função de upload e processamento
audio_data, audio_sample_rate, audio_path = upload_and_process_audio()
if audio_data is None:
    st.stop()  # Interrompe se nenhum áudio foi carregado.
# Função para carregar o modelo YAMNet
@st.cache_resource
def load_yamnet_model():
    """
    Carrega o modelo YAMNet a partir do TensorFlow Hub.
    """
    yam_model = hub.load('https://tfhub.dev/google/yamnet/1')
    return yam_model

# Função para extrair embeddings do áudio com YAMNet
def extract_yamnet_embeddings(yamnet_model, audio_path):
    """
    Extrai embeddings do áudio usando o modelo YAMNet.
    Retorna a classe predita e os embeddings médios das frames.
    """
    basename_audio = os.path.basename(audio_path)
    try:
        # Ler o arquivo de áudio
        sr_orig, wav_data = wavfile.read(audio_path)
        st.write(f"Processando {basename_audio}: Sample Rate = {sr_orig}, Shape = {wav_data.shape}, Dtype = {wav_data.dtype}")
        
        # Converter para mono, se necessário
        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)
            st.write(f"Convertido para mono: Shape = {wav_data.shape}")

        # Normalizar dados
        if wav_data.dtype.kind == 'i':  # Dados inteiros
            max_val = np.iinfo(wav_data.dtype).max
            waveform = wav_data / max_val
            st.write(f"Normalizado de inteiros para float: max_val = {max_val}")
        elif wav_data.dtype.kind == 'f':  # Dados float
            waveform = wav_data
            if np.max(waveform) > 1.0 or np.min(waveform) < -1.0:
                waveform = waveform / np.max(np.abs(waveform))
                st.write("Normalizado para o intervalo [-1.0, 1.0]")
        else:
            raise ValueError(f"Tipo de dado do áudio não suportado: {wav_data.dtype}")
        
        # Garantir taxa de amostragem correta
        sr, waveform = ensure_sample_rate(sr_orig, waveform)
        st.write(f"Sample Rate ajustado: {sr}")

        # Garantir que o waveform está em float32
        waveform = waveform.astype(np.float32)

        # Executar o modelo YAMNet
        scores, embeddings, spectrogram = yamnet_model(waveform)
        st.write(f"Embeddings extraídos: Shape = {embeddings.shape}")

        # Calcular a média dos scores por frame
        scores_np = scores.numpy()
        mean_scores = scores_np.mean(axis=0)
        pred_class = mean_scores.argmax()
        st.write(f"Classe predita pelo YAMNet: {pred_class}")

        # Calcular a média dos embeddings para obter um embedding fixo
        mean_embedding = embeddings.numpy().mean(axis=0)  # Shape: (1024,)
        st.write(f"Média dos embeddings: Shape = {mean_embedding.shape}")

        return pred_class, mean_embedding

    except Exception as e:
        st.error(f"Erro ao processar {basename_audio}: {e}")
        return -1, None

# Carregar o modelo YAMNet
st.header("Carregar e Processar Áudio com YAMNet")
yamnet_model = load_yamnet_model()
st.success("Modelo YAMNet carregado com sucesso.")

# Extrair embeddings do áudio carregado
if audio_path:
    st.subheader("Extração de Embeddings")
    predicted_class, embedding = extract_yamnet_embeddings(yamnet_model, audio_path)

    if embedding is not None:
        st.write("**Embeddings extraídos com sucesso.**")
        st.write(f"**Classe predita:** {predicted_class}")
    else:
        st.error("Falha na extração de embeddings.")
# Função para adicionar ruído ao áudio
def add_noise(waveform, noise_factor=0.005):
    """
    Adiciona ruído branco ao waveform.
    """
    noise = np.random.randn(len(waveform))
    augmented_waveform = waveform + noise_factor * noise
    return augmented_waveform.astype(np.float32)

# Função para esticar o tempo do áudio
def time_stretch(waveform, rate=1.1):
    """
    Estica o tempo do áudio.
    """
    return librosa.effects.time_stretch(waveform, rate)

# Função para mudar o pitch do áudio
def pitch_shift(waveform, sr, n_steps=2):
    """
    Altera o pitch do áudio.
    """
    return librosa.effects.pitch_shift(waveform, sr, n_steps)

# Função para aplicar métodos de data augmentation
def perform_data_augmentation(waveform, sr, augmentation_methods, rate=1.1, n_steps=2):
    """
    Aplica uma lista de métodos de data augmentation ao waveform.
    """
    augmented_waveforms = [waveform]
    for method in augmentation_methods:
        if method == "Add Noise":
            augmented_waveforms.append(add_noise(waveform))
        elif method == "Time Stretch":
            try:
                augmented_waveforms.append(time_stretch(waveform, rate))
            except Exception as e:
                st.warning(f"Erro ao aplicar Time Stretch: {e}")
        elif method == "Pitch Shift":
            try:
                augmented_waveforms.append(pitch_shift(waveform, sr, n_steps))
            except Exception as e:
                st.warning(f"Erro ao aplicar Pitch Shift: {e}")
    return augmented_waveforms

# Interface no Streamlit para configuração de data augmentation
st.header("Data Augmentation (Opcional)")
augment = st.checkbox("Aplicar Data Augmentation")

if augment:
    # Selecionar métodos de data augmentation
    augmentation_methods = st.multiselect(
        "Escolha os métodos de data augmentation:",
        options=["Add Noise", "Time Stretch", "Pitch Shift"],
        default=["Add Noise"]
    )
    rate = st.slider("Fator de estiramento (Time Stretch):", min_value=0.5, max_value=2.0, value=1.1, step=0.1)
    n_steps = st.slider("Mudança de pitch (Pitch Shift, em semitons):", min_value=-12, max_value=12, value=2, step=1)

    # Aplicar data augmentation no áudio carregado
    if audio_path:
        st.subheader("Resultados de Data Augmentation")
        try:
            sr_orig, waveform = wavfile.read(audio_path)
            if waveform.ndim > 1:  # Converter para mono
                waveform = waveform.mean(axis=1)
            sr, waveform = ensure_sample_rate(sr_orig, waveform)

            augmented_waveforms = perform_data_augmentation(
                waveform, sr, augmentation_methods, rate=rate, n_steps=n_steps
            )

            # Exibir os waveforms originais e aumentados
            fig, axes = plt.subplots(len(augmented_waveforms), 1, figsize=(10, 5 * len(augmented_waveforms)))
            if len(augmented_waveforms) == 1:
                axes = [axes]
            for idx, aug_waveform in enumerate(augmented_waveforms):
                axes[idx].plot(aug_waveform)
                axes[idx].set_title(f"Waveform {'Original' if idx == 0 else f'Aumentado {idx}'}")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erro ao aplicar data augmentation: {e}")
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Função para balancear as classes
def balance_classes(X, y, method):
    """
    Balanceia as classes utilizando SMOTE (oversampling) ou undersampling.
    """
    if method == "Oversample":
        smote = SMOTE(random_state=42)
        X_bal, y_bal = smote.fit_resample(X, y)
    elif method == "Undersample":
        rus = RandomUnderSampler(random_state=42)
        X_bal, y_bal = rus.fit_resample(X, y)
    else:
        X_bal, y_bal = X, y
    return X_bal, y_bal

# Interface no Streamlit para configuração do balanceamento
st.header("Balanceamento de Classes (Opcional)")
balance_method = st.selectbox(
    "Selecione o método de balanceamento de classes:",
    options=["None", "Oversample (SMOTE)", "Undersample"],
    index=0
)

if balance_method != "None":
    st.write(f"Aplicando método de balanceamento: {balance_method}")

    # Verificar se embeddings e labels estão prontos
    if 'embeddings' in st.session_state and 'labels' in st.session_state:
        try:
            embeddings = st.session_state['embeddings']
            labels = st.session_state['labels']

            # Aplicar balanceamento
            embeddings_bal, labels_bal = balance_classes(embeddings, labels, balance_method)

            # Atualizar estado do Streamlit
            st.session_state['embeddings_bal'] = embeddings_bal
            st.session_state['labels_bal'] = labels_bal

            # Contar classes após balanceamento
            unique_classes, counts = np.unique(labels_bal, return_counts=True)
            class_counts = dict(zip(unique_classes, counts))

            st.write("**Distribuição de classes após balanceamento:**")
            st.write(pd.DataFrame({"Classe": list(class_counts.keys()), "Quantidade": list(class_counts.values())}))
        except Exception as e:
            st.error(f"Erro ao aplicar balanceamento de classes: {e}")
    else:
        st.warning("Os embeddings e labels ainda não estão disponíveis. Certifique-se de ter preparado os dados anteriormente.")
else:
    st.write("Nenhum método de balanceamento será aplicado.")
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Definir o modelo de classificador
class AudioClassifier(nn.Module):
    """
    Modelo simples de classificação de áudio com PyTorch.
    """
    def __init__(self, input_dim, num_classes):
        super(AudioClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Função de treinamento
def train_audio_classifier(X_train, y_train, X_val, y_val, input_dim, num_classes, epochs, learning_rate, batch_size, l2_lambda, patience):
    """
    Treina um classificador simples em PyTorch com early stopping e avaliação.
    """
    # Converter dados para tensores
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Criar DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instanciar o modelo
    classifier = AudioClassifier(input_dim, num_classes).to(device)

    # Configurar função de perda e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    # Variáveis para early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None

    # Listas para armazenar métricas
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    progress_bar = st.progress(0)
    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0
        running_corrects = 0

        # Treinamento
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # Cálculo de métricas de treinamento
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        # Validação
        classifier.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = classifier(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        # Cálculo de métricas de validação
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc.item())

        # Atualizar barra de progresso
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)

        # Exibir métricas
        st.write(f"**Época {epoch+1}/{epochs}**")
        st.write(f"Treino - Perda: {epoch_loss:.4f}, Acurácia: {epoch_acc.item():.4f}")
        st.write(f"Validação - Perda: {val_epoch_loss:.4f}, Acurácia: {val_epoch_acc.item():.4f}")

        # Early stopping
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
            best_model_wts = classifier.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                st.write("Parada antecipada ativada.")
                break

    # Carregar os melhores pesos do modelo
    if best_model_wts is not None:
        classifier.load_state_dict(best_model_wts)

    # Salvar o modelo e as métricas no estado do Streamlit
    st.session_state['classifier'] = classifier
    st.session_state['train_metrics'] = (train_losses, train_accuracies)
    st.session_state['val_metrics'] = (val_losses, val_accuracies)

    return classifier

# Interface no Streamlit para treinamento
st.header("Treinamento do Classificador")

# Verificar se embeddings e labels estão disponíveis
if 'embeddings_bal' in st.session_state and 'labels_bal' in st.session_state:
    embeddings = st.session_state['embeddings_bal']
    labels = st.session_state['labels_bal']

    # Dividir dados em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Configurar parâmetros do modelo
    input_dim = embeddings.shape[1]
    num_classes = len(set(labels))

    # Lidar com parâmetros configurados pelo usuário na barra lateral
    epochs = st.sidebar.slider("Número de Épocas", min_value=10, max_value=100, value=30, step=5)
    learning_rate = st.sidebar.select_slider("Taxa de Aprendizagem", options=[0.1, 0.01, 0.001, 0.0001], value=0.001)
    batch_size = st.sidebar.selectbox("Tamanho de Lote", options=[8, 16, 32, 64], index=1)
    l2_lambda = st.sidebar.number_input("Regularização L2 (Weight Decay)", min_value=0.0, max_value=0.1, value=0.01, step=0.01)
    patience = st.sidebar.number_input("Paciência para Early Stopping", min_value=1, max_value=10, value=3, step=1)

    # Treinamento
    classifier = train_audio_classifier(
        X_train, y_train, X_val, y_val, input_dim, num_classes, 
        epochs, learning_rate, batch_size, l2_lambda, patience
    )
    st.success("Treinamento concluído.")
else:
    st.warning("Os dados balanceados ainda não estão disponíveis. Certifique-se de ter completado o balanceamento no módulo anterior.")
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plota as métricas de treinamento (perda e acurácia).
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Perda
    ax[0].plot(range(1, len(train_losses)+1), train_losses, label='Treino')
    ax[0].plot(range(1, len(val_losses)+1), val_losses, label='Validação')
    ax[0].set_title('Perda por Época')
    ax[0].set_xlabel('Épocas')
    ax[0].set_ylabel('Perda')
    ax[0].legend()

    # Acurácia
    ax[1].plot(range(1, len(train_accuracies)+1), train_accuracies, label='Treino')
    ax[1].plot(range(1, len(val_accuracies)+1), val_accuracies, label='Validação')
    ax[1].set_title('Acurácia por Época')
    ax[1].set_xlabel('Épocas')
    ax[1].set_ylabel('Acurácia')
    ax[1].legend()

    st.pyplot(fig)
    plt.close(fig)

def evaluate_classifier(classifier, X_val, y_val, classes):
    """
    Avalia o classificador nos dados de validação e exibe métricas e gráficos.
    """
    classifier.eval()
    all_preds = []
    all_labels = []

    # Converter os dados para tensores
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = classifier(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Relatório de Classificação
    st.write("### Relatório de Classificação")
    target_names = [f"Classe {cls}" for cls in classes]
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    # Matriz de Confusão
    st.write("### Matriz de Confusão")
    cm = confusion_matrix(all_labels, all_preds)
    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matriz de Confusão')
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # Curva ROC e AUC
    st.write("### Curva ROC")
    if len(classes) > 2:
        # Para múltiplas classes, utilizamos One-vs-Rest
        from sklearn.preprocessing import label_binarize

        y_test_binarized = label_binarize(all_labels, classes=range(len(classes)))
        y_pred_binarized = label_binarize(all_preds, classes=range(len(classes)))

        fpr = dict()
        tpr = dict()
        roc_auc_dict = dict()
        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
            roc_auc_dict[i] = auc(fpr[i], tpr[i])

        # Plotar Curva ROC para cada classe
        fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
        colors = sns.color_palette("hsv", len(classes))
        for i, color in zip(range(len(classes)), colors):
            ax_roc.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'Classe {i} (AUC = {roc_auc_dict[i]:0.2f})')

        ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Curva ROC (One-vs-Rest)')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
        plt.close(fig_roc)
    else:
        fpr, tpr, _ = roc_curve(all_labels, [p[1] for p in all_preds])
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
        ax_roc.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Curva ROC')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    # F1-Score
    st.write("### F1-Score por Classe")
    f1_scores = f1_score(all_labels, all_preds, average=None)
    f1_df = pd.DataFrame({'Classe': target_names, 'F1-Score': f1_scores})
    st.write(f1_df)
def classify_new_audio(uploaded_audio, classifier, yamnet_model, classes):
    """
    Classifica um novo arquivo de áudio carregado pelo usuário.
    Exibe a classe predita, a confiança e visualizações do áudio.
    """
    try:
        # Salvar o arquivo de áudio em um diretório temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmp_audio:
            tmp_audio.write(uploaded_audio.read())
            tmp_audio_path = tmp_audio.name

        # Audição do Áudio Carregado
        try:
            sample_rate, wav_data = wavfile.read(tmp_audio_path)
            # Verificar se o áudio é mono e 16kHz
            if wav_data.ndim > 1:
                wav_data = wav_data.mean(axis=1)
            if sample_rate != 16000:
                sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
            # Normalizar
            if wav_data.dtype.kind == 'i':
                wav_data = wav_data / np.iinfo(wav_data.dtype).max
            elif wav_data.dtype.kind == 'f':
                if np.max(wav_data) > 1.0 or np.min(wav_data) < -1.0:
                    wav_data = wav_data / np.max(np.abs(wav_data))
            # Convert to float32
            wav_data = wav_data.astype(np.float32)
            duration = len(wav_data) / sample_rate
            st.write(f"**Sample rate:** {sample_rate} Hz")
            st.write(f"**Total duration:** {duration:.2f}s")
            st.audio(wav_data, format='audio/wav', sample_rate=sample_rate)
        except Exception as e:
            st.error(f"Erro ao processar o áudio para audição: {e}")
            wav_data = None

        # Extrair embeddings
        pred_class, embedding = extract_yamnet_embeddings(yamnet_model, tmp_audio_path)

        if embedding is not None and pred_class != -1:
            # Converter o embedding para tensor
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)

            # Classificar
            classifier.eval()
            with torch.no_grad():
                output = classifier(embedding_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                class_idx = predicted.item()
                class_name = classes[class_idx]
                confidence_score = confidence.item()

            # Exibir resultados
            st.write(f"**Classe Predita:** {class_name}")
            st.write(f"**Confiança:** {confidence_score:.4f}")

            # Visualização do Áudio
            st.subheader("Visualização do Áudio")
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))

            # Plot da Forma de Onda
            ax[0].plot(wav_data)
            ax[0].set_title("Forma de Onda")
            ax[0].set_xlabel("Amostras")
            ax[0].set_ylabel("Amplitude")

            # Plot do Espectrograma
            S = librosa.feature.melspectrogram(y=wav_data, sr=sample_rate, n_mels=128)
            S_DB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_DB, sr=sample_rate, x_axis='time', y_axis='mel', ax=ax[1])
            fig.colorbar(ax[1].collections[0], ax=ax[1], format='%+2.0f dB')
            ax[1].set_title("Espectrograma Mel")

            st.pyplot(fig)
            plt.close(fig)
        else:
            st.error("Não foi possível classificar o áudio.")
    except Exception as e:
        st.error(f"Erro ao classificar o áudio: {e}")
    finally:
        # Remover arquivos temporários
        try:
            os.remove(tmp_audio_path)
        except Exception as e:
            st.warning(f"Erro ao remover arquivos temporários: {e}")
def detect_pitch_and_generate_score(uploaded_audio):
    """
    Realiza a detecção de pitch no áudio carregado e converte em uma partitura.
    """
    try:
        # Salvar o arquivo de áudio em um diretório temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmp_audio:
            tmp_audio.write(uploaded_audio.read())
            tmp_audio_path = tmp_audio.name

        # Carregar o modelo SPICE
        spice_model = hub.load("https://tfhub.dev/google/spice/2")

        # Processar o áudio
        sample_rate, wav_data = wavfile.read(tmp_audio_path)
        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)
        if sample_rate != 16000:
            sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

        # Normalizar
        wav_data = wav_data / np.max(np.abs(wav_data)) if np.max(np.abs(wav_data)) != 0 else wav_data

        # Detecção de pitch
        model_output = spice_model.signatures["serving_default"](tf.constant(wav_data, tf.float32))
        pitch_outputs = model_output["pitch"].numpy().flatten()
        uncertainty_outputs = model_output["uncertainty"].numpy().flatten()

        # Confiança nos pitches detectados
        confidence_outputs = 1.0 - uncertainty_outputs
        confident_indices = np.where(confidence_outputs >= 0.9)[0]
        confident_pitches = pitch_outputs[confidence_outputs >= 0.9]
        confident_pitch_values_hz = [output2hz(p) for p in confident_pitches]

        # Conversão para notas musicais
        st.subheader("Conversão para Notas Musicais")
        A4 = 440
        C0 = A4 * pow(2, -4.75)
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        def hz2offset(freq):
            if freq == 0:
                return None
            h = round(12 * math.log2(freq / C0))
            return 12 * math.log2(freq / C0) - h

        offsets = [hz2offset(p) for p in confident_pitch_values_hz if p != 0]
        ideal_offset = statistics.mean(offsets) if offsets else 0.0

        def quantize_predictions(group, ideal_offset):
            non_zero_values = [v for v in group if v != 0]
            zero_values_count = len(group) - len(non_zero_values)

            if zero_values_count > 0.8 * len(group):
                return 0.51 * len(non_zero_values), "Rest"
            else:
                h = round(statistics.mean([
                    12 * math.log2(freq / C0) - ideal_offset for freq in non_zero_values
                ]))
                octave = h // 12
                n = h % 12
                note = note_names[n] + str(octave)
                error = sum([
                    abs(12 * math.log2(freq / C0) - ideal_offset - h)
                    for freq in non_zero_values
                ])
                return error, note

        predictions_per_eighth = 20
        prediction_start_offset = 0

        def get_quantization_and_error(pitch_outputs_and_rests, predictions_per_eighth, prediction_start_offset, ideal_offset):
            pitch_outputs_and_rests = [0] * prediction_start_offset + list(pitch_outputs_and_rests)
            groups = [
                pitch_outputs_and_rests[i:i + predictions_per_eighth]
                for i in range(0, len(pitch_outputs_and_rests), predictions_per_eighth)
            ]

            quantization_error = 0
            notes_and_rests = []
            for group in groups:
                error, note_or_rest = quantize_predictions(group, ideal_offset)
                quantization_error += error
                notes_and_rests.append(note_or_rest)

            return quantization_error, notes_and_rests

        best_error = float("inf")
        best_notes_and_rests = None

        for ppn in range(15, 35, 1):
            for pso in range(ppn):
                error, notes_and_rests = get_quantization_and_error(
                    confident_pitch_values_hz, ppn, pso, ideal_offset
                )
                if error < best_error:
                    best_error = error
                    best_notes_and_rests = notes_and_rests

        # Remover rests iniciais e finais
        while best_notes_and_rests and best_notes_and_rests[0] == 'Rest':
            best_notes_and_rests = best_notes_and_rests[1:]
        while best_notes_and_rests and best_notes_and_rests[-1] == 'Rest':
            best_notes_and_rests = best_notes_and_rests[:-1]

        st.write(f"Notas e Rests Detectados: {best_notes_and_rests}")

        # Criar partitura com music21
        if best_notes_and_rests:
            sc = music21.stream.Score()
            part = music21.stream.Part()
            sc.insert(0, part)

            bpm = 60  # Ajustar conforme necessário
            metronome = music21.tempo.MetronomeMark(number=bpm)
            part.insert(0, metronome)

            for snote in best_notes_and_rests:
                if snote == 'Rest':
                    part.append(music21.note.Rest(type='quarter'))
                else:
                    try:
                        part.append(music21.note.Note(snote, type='quarter'))
                    except music21.pitch.PitchException:
                        st.warning(f"Nota inválida: {snote}")

            if sc.isWellFormedNotation():
                showScore(sc)

                midi_file = tmp_audio_path[:-4] + ".mid"
                sc.write("midi", fp=midi_file)

                with open(midi_file, 'rb') as f:
                    midi_data = f.read()
                st.download_button(
                    label="Download do Arquivo MIDI",
                    data=midi_data,
                    file_name=os.path.basename(midi_file),
                    mime="audio/midi"
                )
            else:
                st.error("A partitura criada não está bem-formada.")
    except Exception as e:
        st.error(f"Erro ao processar o áudio para detecção de pitch: {e}")
def test_and_feedback(uploaded_audio):
    """
    Permite que o usuário teste a aplicação e forneça feedback.
    """
    st.header("Teste e Feedback")
    st.write("""
    **Como usar:**
    1. Faça o upload de um novo arquivo de áudio.
    2. A aplicação classificará o áudio, detectará notas musicais e gerará uma partitura.
    3. Ofereça seu feedback para melhoria contínua.
    """)

    if uploaded_audio is not None:
        try:
            # Classificar e gerar partitura
            detect_pitch_and_generate_score(uploaded_audio)

            # Solicitar feedback
            st.subheader("Feedback")
            st.write("""
            Por favor, forneça seu feedback sobre o resultado da classificação e geração de partitura:
            """)

            # Caixa de texto para feedback
            feedback = st.text_area("Digite seu feedback aqui:")
            if st.button("Enviar Feedback"):
                if feedback.strip():
                    # Salvar feedback para análise posterior
                    with open("feedback_logs.txt", "a") as feedback_file:
                        feedback_file.write(f"Feedback: {feedback}\n")
                    st.success("Obrigado pelo seu feedback!")
                else:
                    st.warning("Por favor, digite algum feedback antes de enviar.")
        except Exception as e:
            st.error(f"Erro ao processar o áudio: {e}")

# Chamando o módulo no fluxo principal
uploaded_audio_for_test = st.file_uploader(
    "Faça upload de um arquivo de áudio para teste", type=["wav", "mp3", "ogg", "flac"]
)
if uploaded_audio_for_test is not None:
    test_and_feedback(uploaded_audio_for_test)

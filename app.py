import os
import zipfile
import shutil
import tempfile
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import streamlit as st
import gc
import logging
import io
import tensorflow as tf
import tensorflow_hub as hub
from scipy.io import wavfile
import scipy.signal
from datetime import datetime
import librosa
from IPython.display import Audio as IPyAudio  # Para audição no Jupyter (se necessário)

# Suprimir avisos relacionados ao torch.classes
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.classes.*")

# Definir o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurações para gráficos mais bonitos
sns.set_style('whitegrid')

# Definir seed para reprodutibilidade
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Garantir reprodutibilidade
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Definir a seed

@st.cache_resource
def load_yamnet_model():
    """
    Carrega o modelo YAMNet do TF Hub.
    """
    yam_model = hub.load('https://tfhub.dev/google/yamnet/1')
    return yam_model

def ensure_sample_rate(original_sr, waveform, desired_sr=16000):
    """
    Resample se não estiver em 16 kHz.
    """
    if original_sr != desired_sr:
        desired_length = int(round(float(len(waveform)) / original_sr * desired_sr))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sr, waveform

def add_noise(waveform, noise_factor=0.005):
    """
    Adiciona ruído branco ao áudio.
    """
    noise = np.random.randn(len(waveform))
    augmented_waveform = waveform + noise_factor * noise
    augmented_waveform = augmented_waveform.astype(np.float32)
    return augmented_waveform

def time_stretch(waveform, rate=1.1):
    """
    Estica o tempo do áudio.
    """
    return librosa.effects.time_stretch(waveform, rate)

def pitch_shift(waveform, sr, n_steps=2):
    """
    Muda a altura do áudio.
    """
    return librosa.effects.pitch_shift(waveform, sr, n_steps)

def perform_data_augmentation(waveform, sr, augmentation_methods, rate=1.1, n_steps=2):
    """
    Aplica data augmentation no áudio.
    """
    augmented_waveforms = [waveform]
    for method in augmentation_methods:
        if method == 'Add Noise':
            augmented_waveforms.append(add_noise(waveform))
        elif method == 'Time Stretch':
            try:
                stretched = time_stretch(waveform, rate=rate)
                augmented_waveforms.append(stretched)
            except Exception as e:
                st.warning(f"Erro ao aplicar Time Stretch: {e}")
        elif method == 'Pitch Shift':
            try:
                shifted = pitch_shift(waveform, sr, n_steps=n_steps)
                augmented_waveforms.append(shifted)
            except Exception as e:
                st.warning(f"Erro ao aplicar Pitch Shift: {e}")
    return augmented_waveforms

def extract_yamnet_embeddings(yamnet_model, audio_path):
    """
    Extrai embeddings usando o modelo YAMNet para um arquivo de áudio.
    Retorna a classe predita e a média dos embeddings das frames.
    """
    basename_audio = os.path.basename(audio_path)
    try:
        sr_orig, wav_data = wavfile.read(audio_path)
        st.write(f"Processando {basename_audio}: Sample Rate = {sr_orig}, Shape = {wav_data.shape}, Dtype = {wav_data.dtype}")
        
        # Verificar se está estéreo
        if wav_data.ndim > 1:
            # Converter para mono
            wav_data = wav_data.mean(axis=1)
            st.write(f"Convertido para mono: Shape = {wav_data.shape}")
        
        # Normalizar para [-1, 1] ou verificar se já está normalizado
        if wav_data.dtype.kind == 'i':
            # Dados inteiros
            max_val = np.iinfo(wav_data.dtype).max
            waveform = wav_data / max_val
            st.write(f"Normalizado de inteiros para float: max_val = {max_val}")
        elif wav_data.dtype.kind == 'f':
            # Dados float
            waveform = wav_data
            # Verificar se os dados estão fora do intervalo [-1.0, 1.0]
            if np.max(waveform) > 1.0 or np.min(waveform) < -1.0:
                waveform = waveform / np.max(np.abs(waveform))
                st.write("Normalizado para o intervalo [-1.0, 1.0]")
        else:
            raise ValueError(f"Tipo de dado do áudio não suportado: {wav_data.dtype}")
    
        # Garantir que é float32
        waveform = waveform.astype(np.float32)
    
        # Ajustar sample rate
        sr, waveform = ensure_sample_rate(sr_orig, waveform)
        st.write(f"Sample Rate ajustado: {sr}")
    
        # Executar o modelo YAMNet
        # yamnet_model retorna: scores, embeddings, spectrogram
        scores, embeddings, spectrogram = yamnet_model(waveform)
        st.write(f"Embeddings extraídos: Shape = {embeddings.shape}")
    
        # scores.shape = [frames, 521]
        scores_np = scores.numpy()
        mean_scores = scores_np.mean(axis=0)  # média por frame
        pred_class = mean_scores.argmax()
        st.write(f"Classe predita pelo YAMNet: {pred_class}")
    
        # Calcular a média dos embeddings das frames para obter um embedding fixo
        mean_embedding = embeddings.numpy().mean(axis=0)  # Shape: (1024,)
        st.write(f"Média dos embeddings das frames: Shape = {mean_embedding.shape}")
    
        return pred_class, mean_embedding
    except Exception as e:
        st.error(f"Erro ao processar {basename_audio}: {e}")
        return -1, None

def balance_classes(X, y, method):
    """
    Balanceia as classes usando oversampling ou undersampling.
    """
    if method == 'Oversample':
        smote = SMOTE(random_state=42)
        X_bal, y_bal = smote.fit_resample(X, y)
    elif method == 'Undersample':
        rus = RandomUnderSampler(random_state=42)
        X_bal, y_bal = rus.fit_resample(X, y)
    else:
        X_bal, y_bal = X, y
    return X_bal, y_bal

def train_audio_classifier(X_train, y_train, X_val, y_val, input_dim, num_classes, epochs, learning_rate, batch_size, l2_lambda, patience):
    """
    Treina um classificador simples em PyTorch com os embeddings extraídos.
    """
    # Converter para tensores
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Criar DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Definir um classificador simples
    class AudioClassifier(nn.Module):
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

    classifier = AudioClassifier(input_dim, num_classes).to(device)

    # Definir a função de perda e otimizador com L2 regularization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    # Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None

    # Inicializar listas para métricas
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    progress_bar = st.progress(0)
    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0
        running_corrects = 0

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

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc.item())

        # Atualizar a barra de progresso
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)

        # Exibir métricas do epoch
        st.write(f"### Época {epoch+1}/{epochs}")
        st.write(f"**Treino - Perda:** {epoch_loss:.4f}, **Acurácia:** {epoch_acc:.4f}")
        st.write(f"**Validação - Perda:** {val_epoch_loss:.4f}, **Acurácia:** {val_epoch_acc:.4f}")

        # Early Stopping
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
            best_model_wts = classifier.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                st.write("Early stopping!")
                break

    # Carregar os melhores pesos do modelo se houver
    if best_model_wts is not None:
        classifier.load_state_dict(best_model_wts)

    # Plotar métricas
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

    # Avaliação no conjunto de validação
    classifier.eval()
    all_preds = []
    all_labels = []

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
    target_names = [f"Classe {cls}" for cls in set(y_train)]
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    return classifier

def main():
    # Configurações da página - Deve ser chamado antes de qualquer outro comando do Streamlit
    st.set_page_config(page_title="Classificação de Áudio com YAMNet", layout="wide")
    st.title("Classificação de Áudio com YAMNet")
    st.write("Este aplicativo permite treinar um classificador de áudio supervisionado utilizando o modelo YAMNet para extrair embeddings.")

    # Sidebar para parâmetros de treinamento e pré-processamento
    st.sidebar.header("Configurações")
    
    # Parâmetros de Treinamento
    st.sidebar.subheader("Parâmetros de Treinamento")
    epochs = st.sidebar.number_input("Número de Épocas:", min_value=1, max_value=500, value=50, step=1)
    learning_rate = st.sidebar.select_slider("Taxa de Aprendizagem:", options=[0.1, 0.01, 0.001, 0.0001], value=0.001)
    batch_size = st.sidebar.selectbox("Tamanho de Lote:", options=[8, 16, 32, 64], index=1)
    l2_lambda = st.sidebar.number_input("Regularização L2 (Weight Decay):", min_value=0.0, max_value=0.1, value=0.01, step=0.01)
    patience = st.sidebar.number_input("Paciência para Early Stopping:", min_value=1, max_value=10, value=3, step=1)

    # Opções de Data Augmentation
    st.sidebar.subheader("Data Augmentation")
    augment = st.sidebar.checkbox("Aplicar Data Augmentation")
    if augment:
        augmentation_methods = st.sidebar.multiselect(
            "Métodos de Data Augmentation:",
            options=["Add Noise", "Time Stretch", "Pitch Shift"],
            default=["Add Noise", "Time Stretch"]
        )
        # Parâmetros adicionais para Data Augmentation
        rate = st.sidebar.slider("Rate para Time Stretch:", min_value=0.5, max_value=2.0, value=1.1, step=0.1)
        n_steps = st.sidebar.slider("N Steps para Pitch Shift:", min_value=-12, max_value=12, value=2, step=1)
    else:
        augmentation_methods = []
        rate = 1.1
        n_steps = 2

    # Opções de Balanceamento de Classes
    st.sidebar.subheader("Balanceamento de Classes")
    balance_method = st.sidebar.selectbox(
        "Método de Balanceamento:",
        options=["None", "Oversample", "Undersample"],
        index=0
    )

    # Seção de Download e Preparação de Arquivos de Áudio
    st.header("Baixando e Preparando Arquivos de Áudio")
    st.write("Você pode baixar um arquivo de áudio de exemplo ou carregar seu próprio arquivo para começar.")

    # Links para Download de Arquivos de Áudio de Exemplo
    st.subheader("Download de Arquivos de Áudio de Exemplo")
    sample_audio_1 = 'speech_whistling2.wav'
    sample_audio_2 = 'miaow_16k.wav'
    sample_audio_1_url = "https://storage.googleapis.com/audioset/speech_whistling2.wav"
    sample_audio_2_url = "https://storage.googleapis.com/audioset/miaow_16k.wav"

    # Função para Baixar Arquivos
    def download_audio(url, filename):
        try:
            import requests
            response = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(response.content)
            st.success(f"Arquivo `{filename}` baixado com sucesso.")
        except Exception as e:
            st.error(f"Erro ao baixar {filename}: {e}")

    # Botões de Download
    if st.button(f"Baixar {sample_audio_1}"):
        download_audio(sample_audio_1_url, sample_audio_1)
    
    if st.button(f"Baixar {sample_audio_2}"):
        download_audio(sample_audio_2_url, sample_audio_2)

    # Audição de Arquivos de Áudio de Exemplo
    st.subheader("Audição de Arquivos de Áudio de Exemplo")
    uploaded_file_example = st.selectbox("Selecione um arquivo de áudio de exemplo para ouvir:", options=["None", sample_audio_1, sample_audio_2])

    if uploaded_file_example != "None" and os.path.exists(uploaded_file_example):
        try:
            sample_rate, wav_data = wavfile.read(uploaded_file_example, 'rb')
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
            st.write(f"**Size of the input:** {len(wav_data)} samples")
            st.audio(wav_data, format='audio/wav', sample_rate=sample_rate)
        except Exception as e:
            st.error(f"Erro ao processar o arquivo de áudio: {e}")
    elif uploaded_file_example != "None":
        st.warning("Arquivo de áudio não encontrado. Por favor, baixe o arquivo antes de tentar ouvir.")

    # Upload de dados supervisionados
    st.header("Upload de Dados Supervisionados")
    st.write("Envie um arquivo ZIP contendo subpastas com arquivos de áudio organizados por classe. Por exemplo:")
    st.write("""
    ```
    dados/
        agua_quente/
            audio1.wav
            audio2.wav
        agua_gelada/
            audio3.wav
            audio4.wav
    ```
    """)
    uploaded_zip = st.file_uploader("Faça upload do arquivo ZIP com os dados de áudio supervisionados", type=["zip"])

    if uploaded_zip is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "uploaded.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.read())
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(tmpdir)
                st.success("Arquivo ZIP extraído com sucesso.")
            except zipfile.BadZipFile:
                st.error("O arquivo enviado não é um ZIP válido.")
                st.stop()

            # Verificar estrutura de diretórios
            classes = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
            if len(classes) < 2:
                st.error("O arquivo ZIP deve conter pelo menos duas subpastas, cada uma representando uma classe.")
                st.stop()
            else:
                st.success(f"Classes encontradas: {classes}")
                # Contar arquivos por classe
                class_counts = {}
                for cls in classes:
                    cls_dir = os.path.join(tmpdir, cls)
                    files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
                    class_counts[cls] = len(files)
                st.write("**Contagem de arquivos por classe:**")
                st.write(class_counts)

                # Preparar dados para treinamento
                st.header("Preparando Dados para Treinamento")
                yamnet_model = load_yamnet_model()
                st.write("Modelo YAMNet carregado.")

                embeddings = []
                labels = []
                label_mapping = {cls: idx for idx, cls in enumerate(classes)}

                total_files = sum(class_counts.values())
                processed_files = 0
                progress_bar = st.progress(0)

                for cls in classes:
                    cls_dir = os.path.join(tmpdir, cls)
                    audio_files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
                    for audio_file in audio_files:
                        pred_class, embedding = extract_yamnet_embeddings(yamnet_model, audio_file)
                        if embedding is not None:
                            if augment:
                                # Carregar o áudio usando librosa para aplicar augmentations
                                try:
                                    waveform, sr = librosa.load(audio_file, sr=16000, mono=True)
                                    augmented_waveforms = perform_data_augmentation(waveform, sr, augmentation_methods, rate=rate, n_steps=n_steps)
                                    for aug_waveform in augmented_waveforms:
                                        # Salvar temporariamente o áudio aumentado para processar
                                        temp_audio_path = os.path.join(tmpdir, "temp_aug.wav")
                                        wavfile.write(temp_audio_path, sr, (aug_waveform * 32767).astype(np.int16))
                                        _, aug_embedding = extract_yamnet_embeddings(yamnet_model, temp_audio_path)
                                        if aug_embedding is not None:
                                            embeddings.append(aug_embedding)
                                            labels.append(label_mapping[cls])
                                        os.remove(temp_audio_path)
                            else:
                                embeddings.append(embedding)
                                labels.append(label_mapping[cls])
                        processed_files += 1
                        progress_bar.progress(processed_files / total_files)

                # Verificações após a extração
                if len(embeddings) == 0:
                    st.error("Nenhum embedding foi extraído. Verifique se os arquivos de áudio estão no formato correto e se o YAMNet está funcionando corretamente.")
                    st.stop()

                # Verificar se todos os embeddings têm o mesmo tamanho
                embedding_shapes = [emb.shape for emb in embeddings]
                unique_shapes = set(embedding_shapes)
                if len(unique_shapes) != 1:
                    st.error(f"Embeddings têm tamanhos inconsistentes: {unique_shapes}")
                    st.stop()

                # Converter para array NumPy
                try:
                    embeddings = np.array(embeddings)
                    labels = np.array(labels)
                    st.write(f"Embeddings convertidos para array NumPy: Shape = {embeddings.shape}")
                except ValueError as ve:
                    st.error(f"Erro ao converter embeddings para array NumPy: {ve}")
                    st.stop()

                # Balanceamento de Classes
                if balance_method != "None":
                    st.write(f"Aplicando balanceamento de classes: {balance_method}")
                    embeddings_bal, labels_bal = balance_classes(embeddings, labels, balance_method)
                    # Contar novamente após balanceamento
                    balanced_counts = {cls: 0 for cls in classes}
                    for label in labels_bal:
                        cls = [k for k, v in label_mapping.items() if v == label][0]
                        balanced_counts[cls] += 1
                    st.write(f"**Contagem de classes após balanceamento:**")
                    st.write(balanced_counts)
                else:
                    embeddings_bal, labels_bal = embeddings, labels

                # Dividir os dados em treino e validação
                try:
                    X_train, X_val, y_train, y_val = train_test_split(
                        embeddings_bal, labels_bal, test_size=0.2, random_state=42, stratify=labels_bal
                    )
                    st.write(f"Dados divididos em treino ({len(X_train)} amostras) e validação ({len(X_val)} amostras).")
                except ValueError as ve:
                    st.error(f"Erro ao dividir os dados: {ve}")
                    st.stop()

                # Treinar o classificador
                st.header("Treinamento do Classificador")
                classifier = train_audio_classifier(
                    X_train, y_train, X_val, y_val, 
                    input_dim=embeddings.shape[1], 
                    num_classes=len(classes), 
                    epochs=epochs, 
                    learning_rate=learning_rate, 
                    batch_size=batch_size, 
                    l2_lambda=l2_lambda, 
                    patience=patience
                )
                st.success("Treinamento do classificador concluído.")

                # Salvar o classificador no estado do Streamlit
                st.session_state['classifier'] = classifier
                st.session_state['classes'] = classes

                # Opção para download do modelo treinado
                buffer = io.BytesIO()
                torch.save(classifier.state_dict(), buffer)
                buffer.seek(0)
                st.download_button(
                    label="Download do Modelo Treinado",
                    data=buffer,
                    file_name="audio_classifier.pth",
                    mime="application/octet-stream"
                )

                # Opção para download do mapeamento de classes
                class_mapping = "\n".join([f"{cls}:{idx}" for cls, idx in label_mapping.items()])
                st.download_button(
                    label="Download do Mapeamento de Classes",
                    data=class_mapping,
                    file_name="classes_mapping.txt",
                    mime="text/plain"
                )

    # Classificação de Novo Áudio
    if 'classifier' in st.session_state and 'classes' in st.session_state:
        st.header("Classificação de Novo Áudio")
        st.write("Envie um arquivo de áudio para ser classificado pelo modelo treinado.")
        uploaded_audio = st.file_uploader("Faça upload do arquivo de áudio para classificação", type=["wav", "mp3", "ogg", "flac"])

        if uploaded_audio is not None:
            try:
                # Salvar o arquivo de áudio em um diretório temporário
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmp_audio:
                    tmp_audio.write(uploaded_audio.read())
                    tmp_audio_path = tmp_audio.name

                # Audição do Áudio Carregado
                try:
                    sample_rate, wav_data = wavfile.read(tmp_audio_path, 'rb')
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
                    st.write(f"**Size of the input:** {len(wav_data)} samples")
                    st.audio(wav_data, format='audio/wav', sample_rate=sample_rate)
                except Exception as e:
                    st.error(f"Erro ao processar o áudio para audição: {e}")
                    wav_data = None

                # Extrair embeddings
                yamnet_model = load_yamnet_model()
                pred_class, embedding = extract_yamnet_embeddings(yamnet_model, tmp_audio_path)

                if embedding is not None and pred_class != -1:
                    # Obter o modelo treinado
                    classifier = st.session_state['classifier']
                    classes = st.session_state['classes']

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
                    img = librosa.display.specshow(S_DB, sr=sample_rate, x_axis='time', y_axis='mel', ax=ax[1])
                    fig.colorbar(img, ax=ax[1], format='%+2.0f dB')
                    ax[1].set_title("Espectrograma Mel")

                    st.pyplot(fig)
                    plt.close(fig)

            except Exception as e:
                st.error(f"Erro ao processar o áudio: {e}")

            finally:
                # Remover arquivos temporários
                try:
                    os.remove(tmp_audio_path)
                except Exception as e:
                    st.warning(f"Erro ao remover arquivos temporários: {e}")

    # Documentação e Agradecimentos
    st.write("### Documentação dos Procedimentos")
    st.write("""
    1. **Baixando e Preparando Arquivos de Áudio**: Você pode baixar arquivos de áudio de exemplo ou carregar seus próprios arquivos para começar.

    2. **Upload de Dados Supervisionados**: Envie um arquivo ZIP contendo subpastas, onde cada subpasta representa uma classe com seus respectivos arquivos de áudio.

    3. **Data Augmentation**: Se selecionado, aplica métodos de data augmentation como adição de ruído, estiramento de tempo e mudança de pitch nos dados de treinamento. Você pode ajustar os parâmetros `rate` e `n_steps` para controlar a intensidade dessas transformações.

    4. **Balanceamento de Classes**: Se selecionado, aplica métodos de balanceamento como oversampling (SMOTE) ou undersampling para tratar classes desbalanceadas.

    5. **Extração de Embeddings**: Utilizamos o YAMNet para extrair embeddings dos arquivos de áudio enviados.

    6. **Treinamento do Classificador**: Com os embeddings extraídos e após as opções de data augmentation e balanceamento, treinamos um classificador personalizado conforme os parâmetros definidos na barra lateral.

    7. **Classificação de Novo Áudio**: Após o treinamento, você pode enviar um novo arquivo de áudio para ser classificado pelo modelo treinado. O aplicativo exibirá a classe predita, a confiança e visualizará a forma de onda e o espectrograma do áudio carregado.

    **Exemplo de Estrutura de Diretórios para Upload:**
    ```
    dados/
        agua_quente/
            audio1.wav
            audio2.wav
        agua_gelada/
            audio3.wav
            audio4.wav
    ```
    """)

    st.write("### Agradecimentos")
    st.write("""
    Desenvolvido por Marcelo Claro.
    
    - **Contato**: marceloclaro@gmail.com
    - **Whatsapp**: (88)98158-7145
    - **Instagram**: [marceloclaro.geomaker](https://www.instagram.com/marceloclaro.geomaker/)
    """)

if __name__ == "__main__":
    main()

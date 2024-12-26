import os
import zipfile
import shutil
import tempfile
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
import streamlit as st
import gc
import logging
import io
import tensorflow as tf
import tensorflow_hub as hub
from scipy.io import wavfile
import scipy.signal
from datetime import datetime

# Supressão dos avisos relacionados ao torch.classes
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.classes.*")

# Definir o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurações para tornar os gráficos mais bonitos
sns.set_style('whitegrid')

def set_seed(seed):
    """
    Define uma seed para garantir a reprodutibilidade.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # As linhas abaixo são recomendadas para garantir reprodutibilidade
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Definir a seed para reprodutibilidade

@st.cache_resource
def load_yamnet_model():
    """
    Carrega o modelo YAMNet do TF Hub (sem causar pickle do objeto).
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

def extract_yamnet_embeddings(yamnet_model, audio_path):
    """
    Extrai embeddings usando o modelo YAMNet para um arquivo de áudio.
    """
    basename_audio = os.path.basename(audio_path)
    try:
        sr_orig, wav_data = wavfile.read(audio_path)
        # Verificar se está estéreo
        if wav_data.ndim > 1:
            # Converter para mono
            wav_data = wav_data.mean(axis=1)
        # Normalizar para [-1, 1] ou verificar se já está normalizado
        if wav_data.dtype.kind == 'i':
            # Dados inteiros
            max_val = np.iinfo(wav_data.dtype).max
            waveform = wav_data / max_val
        elif wav_data.dtype.kind == 'f':
            # Dados float
            waveform = wav_data
            # Verificar se os dados estão fora do intervalo [-1.0, 1.0]
            if np.max(waveform) > 1.0 or np.min(waveform) < -1.0:
                waveform = waveform / np.max(np.abs(waveform))
        else:
            raise ValueError(f"Tipo de dado do áudio não suportado: {wav_data.dtype}")

        # Garantir que é float32
        waveform = waveform.astype(np.float32)

        # Ajustar sample rate
        sr, waveform = ensure_sample_rate(sr_orig, waveform)

        # Executar o modelo YAMNet
        # yamnet_model retorna: scores, embeddings, spectrogram
        scores, embeddings, spectrogram = yamnet_model(waveform)

        # scores.shape = [frames, 521]
        scores_np = scores.numpy()
        mean_scores = scores_np.mean(axis=0)  # média por frame
        pred_class = mean_scores.argmax()
        return pred_class, embeddings.numpy()
    except Exception as e:
        st.error(f"Erro ao processar {basename_audio}: {e}")
        return -1, None

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

        st.write(f"### Época {epoch+1}/{epochs}")
        st.write(f"Treino - Perda: {epoch_loss:.4f}, Acurácia: {epoch_acc:.4f}")
        st.write(f"Validação - Perda: {val_epoch_loss:.4f}, Acurácia: {val_epoch_acc:.4f}")

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
    report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(num_classes)], output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    return classifier

def main():
    # Configurações da página
    st.set_page_config(page_title="Classificação de Áudio com YAMNet", layout="wide")
    st.title("Classificação de Áudio com YAMNet")
    st.write("Este aplicativo permite treinar um classificador de áudio supervisionado utilizando o modelo YAMNet para extrair embeddings.")

    # Sidebar para parâmetros de treinamento
    st.sidebar.header("Parâmetros de Treinamento")
    epochs = st.sidebar.number_input("Número de Épocas:", min_value=1, max_value=500, value=50, step=1)
    learning_rate = st.sidebar.select_slider("Taxa de Aprendizagem:", options=[0.1, 0.01, 0.001, 0.0001], value=0.001)
    batch_size = st.sidebar.selectbox("Tamanho de Lote:", options=[8, 16, 32, 64], index=1)
    l2_lambda = st.sidebar.number_input("Regularização L2 (Weight Decay):", min_value=0.0, max_value=0.1, value=0.01, step=0.01)
    patience = st.sidebar.number_input("Paciência para Early Stopping:", min_value=1, max_value=10, value=3, step=1)

    # Upload de dados supervisionados
    st.header("Upload de Dados Supervisionados")
    st.write("Envie um arquivo ZIP contendo subpastas com arquivos de áudio organizados por classe. Por exemplo:")
    st.write("""
    ```
    dados/
        agua_quente/
            audio1.wav
            audio2.wav
        agua_fria/
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
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            # Verificar estrutura de diretórios
            classes = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
            if len(classes) < 2:
                st.error("O arquivo ZIP deve conter pelo menos duas subpastas, cada uma representando uma classe.")
            else:
                st.success(f"Classes encontradas: {classes}")
                # Contar arquivos por classe
                class_counts = {cls: len([f for f in os.listdir(os.path.join(tmpdir, cls)) if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]) for cls in classes}
                st.write("**Contagem de arquivos por classe:**")
                st.write(class_counts)

                # Preparar dados para treinamento
                st.header("Preparando Dados para Treinamento")
                yamnet_model = load_yamnet_model()
                st.write("Modelo YAMNet carregado.")

                embeddings = []
                labels = []
                label_mapping = {cls: idx for idx, cls in enumerate(classes)}

                progress_bar = st.progress(0)
                total_files = sum(class_counts.values())
                processed_files = 0

                for cls in classes:
                    cls_dir = os.path.join(tmpdir, cls)
                    audio_files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
                    for audio_file in audio_files:
                        pred_class, embedding = extract_yamnet_embeddings(yamnet_model, audio_file)
                        if embedding is not None:
                            embeddings.append(embedding.flatten())
                            labels.append(label_mapping[cls])
                        processed_files += 1
                        progress_bar.progress(processed_files / total_files)

                embeddings = np.array(embeddings)
                labels = np.array(labels)
                st.success("Extração de embeddings concluída.")

                # Dividir os dados em treino e validação
                X_train, X_val, y_train, y_val = train_test_split(embeddings, labels, test_size=0.2, random_state=42, stratify=labels)
                st.write(f"Dados divididos em treino ({len(X_train)} amostras) e validação ({len(X_val)} amostras).")

                # Treinar o classificador
                input_dim = X_train.shape[1]
                num_classes = len(classes)
                st.header("Treinamento do Classificador")
                classifier = train_audio_classifier(X_train, y_train, X_val, y_val, input_dim, num_classes, epochs, learning_rate, batch_size, l2_lambda, patience)
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

                # Carregar a imagem para exibição
                audio_image = Image.new('RGB', (400, 100), color = (73, 109, 137))
                plt.figure(figsize=(4,1))
                plt.text(0.5, 0.5, 'Áudio Enviado', horizontalalignment='center', verticalalignment='center', fontsize=20, color='white')
                plt.axis('off')
                plt.savefig(tmp_audio_path + '_image.png')
                plt.close()
                st.image(tmp_audio_path + '_image.png', caption='Áudio Enviado', use_column_width=True)
            except Exception as e:
                st.error(f"Erro ao processar o áudio: {e}")
                tmp_audio_path = None

            if tmp_audio_path is not None:
                # Extrair embeddings
                yamnet_model = load_yamnet_model()
                pred_class, embedding = extract_yamnet_embeddings(yamnet_model, tmp_audio_path)

                if embedding is not None and pred_class != -1:
                    # Obter o modelo treinado
                    classifier = st.session_state['classifier']
                    classes = st.session_state['classes']

                    # Converter o embedding para tensor
                    embedding_tensor = torch.tensor(embedding.flatten(), dtype=torch.float32).unsqueeze(0).to(device)

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

                # Remover arquivos temporários
                os.remove(tmp_audio_path)
                if os.path.exists(tmp_audio_path + '_image.png'):
                    os.remove(tmp_audio_path + '_image.png')

    st.write("### Documentação dos Procedimentos")
    st.write("""
    1. **Upload de Dados Supervisionados**: Envie um arquivo ZIP contendo subpastas, onde cada subpasta representa uma classe com seus respectivos arquivos de áudio.

    2. **Extração de Embeddings**: Utilizamos o YAMNet para extrair embeddings dos arquivos de áudio enviados.

    3. **Treinamento do Classificador**: Com os embeddings extraídos, treinamos um classificador personalizado conforme os parâmetros definidos na barra lateral.

    4. **Classificação de Novo Áudio**: Após o treinamento, você pode enviar um novo arquivo de áudio para ser classificado pelo modelo treinado.

    **Exemplo de Estrutura de Diretórios para Upload**:
    ```
    dados/
        agua_quente/
            audio1.wav
            audio2.wav
        agua_fria/
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

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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score
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
import librosa.display
import requests  # Para download de arquivos de áudio
import math
import statistics
import music21  # Importação adicionada
import streamlit.components.v1 as components  # Importação adicionada
import pretty_midi
import soundfile as sf

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

# Função de Pitch Shift Renomeada para Evitar Conflitos
def apply_pitch_shift(waveform, sr, n_steps=2):
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
                shifted = apply_pitch_shift(waveform, sr, n_steps=n_steps)  # Ajuste aqui
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
    Inclui validação cruzada e métricas de avaliação.
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
        st.write(f"**Treino - Perda:** {epoch_loss:.4f}, **Acurácia:** {epoch_acc.item():.4f}")
        st.write(f"**Validação - Perda:** {val_epoch_loss:.4f}, **Acurácia:** {val_epoch_acc.item():.4f}")

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

    # Matriz de Confusão
    st.write("### Matriz de Confusão")
    cm = confusion_matrix(all_labels, all_preds)
    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # Curva ROC e AUC
    st.write("### Curva ROC")
    if num_classes > 1:
        # Para múltiplas classes, utilizamos One-vs-Rest
        from sklearn.preprocessing import label_binarize

        y_test_binarized = label_binarize(all_labels, classes=range(num_classes))
        y_pred_binarized = label_binarize(all_preds, classes=range(num_classes))

        if y_test_binarized.shape[1] > 1:
            fpr = dict()
            tpr = dict()
            roc_auc_dict = dict()
            for i in range(num_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
                roc_auc_dict[i] = auc(fpr[i], tpr[i])

            # Plotar Curva ROC para cada classe
            fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
            colors = sns.color_palette("hsv", num_classes)
            for i, color in zip(range(num_classes), colors):
                ax_roc.plot(fpr[i], tpr[i], color=color, lw=2,
                           label=f'Classe {i} (AUC = {roc_auc_dict[i]:0.2f})')

            ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
            plt.close(fig_roc)
        else:
            st.warning("Curva ROC não disponível para uma única classe após binarização.")
    else:
        st.warning("Curva ROC não disponível para uma única classe.")

    # F1-Score
    st.write("### F1-Score por Classe")
    f1_scores = f1_score(all_labels, all_preds, average=None)
    f1_df = pd.DataFrame({'Classe': target_names, 'F1-Score': f1_scores})
    st.write(f1_df)

    return classifier

def midi_to_wav(midi_path, wav_path, soundfont_path):
    """
    Converte um arquivo MIDI para WAV usando um SoundFont.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        audio_data = midi_data.synthesize(fs=16000, sf2_path=soundfont_path)
        sf.write(wav_path, audio_data, 16000)
        return True
    except Exception as e:
        st.error(f"Erro ao converter MIDI para WAV: {e}")
        return False

def showScore(score):
    """
    Renderiza a partitura musical usando OpenSheetMusicDisplay via componente HTML do Streamlit.
    """
    # Obter o conteúdo MusicXML como string
    xml = score.write('musicxml', fp=None)
    
    # Verificar se 'xml' é realmente uma string
    if not isinstance(xml, str):
        st.error("Falha ao converter a partitura para MusicXML como string.")
        return
    
    # Sanitizar o XML para evitar quebra de JavaScript
    sanitized_xml = xml.replace('\\', '\\\\').replace('`', '\\`')
    
    showMusicXML(sanitized_xml)

def showMusicXML(xml):
    """
    Renderiza a partitura musical usando OpenSheetMusicDisplay via componente HTML do Streamlit.
    """
    DIV_ID = "OSMD_div"
    html_content = f"""
    <div id="{DIV_ID}">Carregando OpenSheetMusicDisplay...</div>
    <script>
    (function() {{
        // Carregar OpenSheetMusicDisplay se ainda não estiver carregado
        if (!window.opensheetmusicdisplay) {{
            var script = document.createElement('script');
            script.src = "https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@0.7.6/build/opensheetmusicdisplay.min.js";
            script.onload = function() {{
                initializeOSMD();
            }};
            document.head.appendChild(script);
        }} else {{
            initializeOSMD();
        }}

        function initializeOSMD() {{
            var osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("{DIV_ID}", {{
                drawingParameters: "compacttight"
            }});
            osmd.load(`{xml}`).then(function() {{
                osmd.render();
            }});
        }}
    }})();
    </script>
    """
    components.html(html_content, height=600)

def create_music_score(best_notes_and_rests, tempo, showScore, tmp_audio_path):
    """
    Cria e renderiza a partitura musical usando music21.
    """
    if best_notes_and_rests:
        sc = music21.stream.Score()
        part = music21.stream.Part()
        sc.insert(0, part)

        # Adicionar a assinatura de tempo (compasso 4/4)
        time_signature = music21.meter.TimeSignature('4/4')
        part.insert(0, time_signature)

        # Adicionar a assinatura de tempo com BPM inferido
        a = music21.tempo.MetronomeMark(number=tempo)
        part.insert(1, a)  # Inserir após a assinatura de tempo

        for snote in best_notes_and_rests:
            if snote == 'Rest':
                note_obj = music21.note.Rest()
                note_obj.duration.type = 'quarter'
                part.append(note_obj)
            else:
                try:
                    note_obj = music21.note.Note(snote)
                    note_obj.duration.type = 'quarter'
                    part.append(note_obj)
                except music21.pitch.PitchException:
                    st.warning(f"Nota inválida detectada: {snote}. Será ignorada.")
                    continue

        # Verificar se a partitura está bem-formada
        if sc.isWellFormedNotation():
            # Exibir a partitura usando OpenSheetMusicDisplay (OSMD)
            showScore(sc)

            # Converter as notas musicais em um arquivo MIDI
            converted_audio_file_as_midi = tmp_audio_path[:-4] + '.mid'
            sc.write('midi', fp=converted_audio_file_as_midi)

            # Converter MIDI para WAV
            soundfont_path = 'Path/para/seu/SoundFont.sf2'  # Atualize com o caminho para o seu SoundFont
            converted_audio_file_as_wav = tmp_audio_path[:-4] + '.wav'
            success = midi_to_wav(converted_audio_file_as_midi, converted_audio_file_as_wav, soundfont_path)

            if success:
                # Oferecer o arquivo WAV para download e reprodução
                try:
                    with open(converted_audio_file_as_wav, 'rb') as f:
                        wav_data = f.read()
                    st.download_button(
                        label="Download do Arquivo WAV",
                        data=wav_data,
                        file_name=os.path.basename(converted_audio_file_as_wav),
                        mime="audio/wav"
                    )
                    st.audio(wav_data, format='audio/wav')
                    st.success("Arquivo WAV gerado, reproduzido e disponível para download.")
                except Exception as e:
                    st.error(f"Erro ao gerar ou reproduzir o arquivo WAV: {e}")
            else:
                st.error("Falha na conversão de MIDI para WAV.")
        else:
            st.error("A partitura criada não está bem-formada. Verifique os dados de entrada.")

def main():
    # Configurações da página - Deve ser chamado antes de qualquer outro comando do Streamlit
    st.set_page_config(page_title="Classificação de Áudio com YAMNet e SPICE", layout="wide")
    st.title("Classificação de Áudio com YAMNet e Detecção de Pitch com SPICE")
    st.write("""
    Este aplicativo permite treinar um classificador de áudio supervisionado utilizando o modelo **YAMNet** para extrair embeddings e o modelo **SPICE** para detecção de pitch, melhorando assim a classificação.
    """)

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
    st.write("""
    Você pode baixar arquivos de áudio de exemplo ou carregar seus próprios arquivos para começar.
    """)

    # Links para Download de Arquivos de Áudio de Exemplo
    st.subheader("Download de Arquivos de Áudio de Exemplo")
    sample_audio_1 = 'speech_whistling2.wav'
    sample_audio_2 = 'miaow_16k.wav'
    sample_audio_1_url = "https://storage.googleapis.com/audioset/speech_whistling2.wav"
    sample_audio_2_url = "https://storage.googleapis.com/audioset/miaow_16k.wav"

    # Função para Baixar Arquivos
    def download_audio(url, filename):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Verifica se a requisição foi bem-sucedida
            with open(filename, 'wb') as f:
                f.write(response.content)
            st.success(f"Arquivo `{filename}` baixado com sucesso.")
        except Exception as e:
            st.error(f"Erro ao baixar {filename}: {e}")

    # Botões de Download
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"Baixar {sample_audio_1}"):
            download_audio(sample_audio_1_url, sample_audio_1)
    with col2:
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
    st.write("""
    Envie um arquivo ZIP contendo subpastas com arquivos de áudio organizados por classe. Por exemplo:
    """)
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
                                        sf.write(temp_audio_path, aug_waveform, sr)
                                        _, aug_embedding = extract_yamnet_embeddings(yamnet_model, temp_audio_path)
                                        if aug_embedding is not None:
                                            embeddings.append(aug_embedding)
                                            labels.append(label_mapping[cls])
                                        os.remove(temp_audio_path)
                                except Exception as e:
                                    st.warning(f"Erro ao aplicar data augmentation no arquivo {audio_file}: {e}")
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
        def output2hz(pitch_output):
            # Constants taken from https://tfhub.dev/google/spice/2
            PT_OFFSET = 25.58
            PT_SLOPE = 63.07
            FMIN = 10.0
            BINS_PER_OCTAVE = 12.0
            cqt_bin = pitch_output * PT_SLOPE + PT_OFFSET
            return FMIN * 2.0 ** (1.0 * cqt_bin / BINS_PER_OCTAVE)

        def quantize_predictions(group, ideal_offset):
            # Group values são ou 0, ou um pitch em Hz.
            non_zero_values = [v for v in group if v != 0]
            zero_values_count = len(group) - len(non_zero_values)

            # Criar um rest se 80% for silencioso, caso contrário, criar uma nota.
            if zero_values_count > 0.8 * len(group):
                # Interpretar como um rest. Contar cada nota descartada como um erro, ponderado um pouco pior que uma nota mal cantada (que 'custaria' 0.5).
                return 0.51 * len(non_zero_values), "Rest"
            else:
                # Interpretar como nota, estimando como média das previsões não-rest.
                h = round(
                    statistics.mean([
                        12 * math.log2(freq / C0) - ideal_offset for freq in non_zero_values
                    ])
                )
                octave = h // 12
                n = h % 12
                if n < 0 or n >= len(note_names):
                    st.warning(f"Índice de nota inválido: {n}. Nota será ignorada.")
                    return float('inf'), "Rest"
                note = note_names[n] + str(octave)
                # Erro de quantização é a diferença total da nota quantizada.
                error = sum([
                    abs(12 * math.log2(freq / C0) - ideal_offset - h)
                    for freq in non_zero_values
                ])
                return error, note

        # Definir constantes
        A4 = 440
        C0 = A4 * pow(2, -4.75)
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        # Agrupar pitches em notas (simplificação: usar uma janela deslizante)
        predictions_per_eighth = 40  # Aumentou de 20 para 40
        prediction_start_offset = 0  # Ajustar conforme necessário

        def get_quantization_and_error(pitch_outputs_and_rests, predictions_per_eighth,
                                       prediction_start_offset, ideal_offset):
            # Aplicar o offset inicial - podemos simplesmente adicionar o offset como rests.
            pitch_outputs_and_rests = [0] * prediction_start_offset + list(pitch_outputs_and_rests)
            # Coletar as previsões para cada nota (ou rest).
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

        def classify_new_audio(uploaded_audio):
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
                    librosa.display.specshow(S_DB, sr=sample_rate, x_axis='time', y_axis='mel', ax=ax[1])
                    fig.colorbar(ax[1].collections[0], ax=ax[1], format='%+2.0f dB')
                    ax[1].set_title("Espectrograma Mel")

                    st.pyplot(fig)
                    plt.close(fig)

                    # Detecção de Pitch com SPICE
                    st.subheader("Detecção de Pitch com SPICE")
                    # Carregar o modelo SPICE
                    spice_model = hub.load("https://tfhub.dev/google/spice/2")

                    # Normalizar as amostras de áudio
                    audio_samples = wav_data / np.max(np.abs(wav_data)) if np.max(np.abs(wav_data)) != 0 else wav_data

                    # Executar o modelo SPICE
                    model_output = spice_model.signatures["serving_default"](tf.constant(audio_samples, tf.float32))

                    pitch_outputs = model_output["pitch"].numpy().flatten()
                    uncertainty_outputs = model_output["uncertainty"].numpy().flatten()

                    # 'Uncertainty' basicamente significa a inversão da confiança.
                    confidence_outputs = 1.0 - uncertainty_outputs

                    # Plotar pitch e confiança
                    fig_pitch, ax_pitch = plt.subplots(figsize=(20, 10))
                    ax_pitch.plot(pitch_outputs, label='Pitch')
                    ax_pitch.plot(confidence_outputs, label='Confiança')
                    ax_pitch.legend(loc="lower right")
                    ax_pitch.set_title("Pitch e Confiança com SPICE")
                    st.pyplot(fig_pitch)
                    plt.close(fig_pitch)

                    # Remover pitches com baixa confiança (ajuste o limiar)
                    confident_indices = np.where(confidence_outputs >= 0.8)[0]  # Reduziu de 0.9 para 0.8
                    confident_pitches = pitch_outputs[confidence_outputs >= 0.8]
                    confident_pitch_values_hz = [output2hz(p) for p in confident_pitches]

                    # Plotar apenas pitches com alta confiança
                    fig_confident_pitch, ax_confident_pitch = plt.subplots(figsize=(20, 10))
                    ax_confident_pitch.scatter(confident_indices, confident_pitch_values_hz, c="r", label='Pitch Confiante')
                    ax_confident_pitch.set_ylim([0, 2000])  # Ajustar conforme necessário
                    ax_confident_pitch.set_title("Pitches Confidentes Detectados com SPICE")
                    ax_confident_pitch.set_xlabel("Amostras")
                    ax_confident_pitch.set_ylabel("Pitch (Hz)")
                    ax_confident_pitch.legend()
                    st.pyplot(fig_confident_pitch)
                    plt.close(fig_confident_pitch)

                    # Conversão de Pitches para Notas Musicais
                    st.subheader("Notas Musicais Detectadas")
                    # Definir constantes para conversão
                    A4 = 440
                    C0 = A4 * pow(2, -4.75)
                    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

                    def hz2offset(freq):
                        # Medir o erro de quantização para uma única nota.
                        if freq == 0:  # Silêncio sempre tem erro zero.
                            return None
                        # Nota quantizada.
                        h = round(12 * math.log2(freq / C0))
                        return 12 * math.log2(freq / C0) - h

                    # Calcular o offset ideal
                    offsets = [hz2offset(p) for p in confident_pitch_values_hz if p != 0]
                    if offsets:
                        ideal_offset = statistics.mean(offsets)
                    else:
                        ideal_offset = 0.0
                    # Garantir que ideal_offset é float
                    ideal_offset = float(ideal_offset)
                    st.write(f"Ideal Offset: {ideal_offset:.4f}")

                    # Função para quantizar previsões
                    def quantize_predictions(group, ideal_offset):
                        # Group values são ou 0, ou um pitch em Hz.
                        non_zero_values = [v for v in group if v != 0]
                        zero_values_count = len(group) - len(non_zero_values)

                        # Criar um rest se 80% for silencioso, caso contrário, criar uma nota.
                        if zero_values_count > 0.8 * len(group):
                            # Interpretar como um rest. Contar cada nota descartada como um erro, ponderado um pouco pior que uma nota mal cantada (que 'custaria' 0.5).
                            return 0.51 * len(non_zero_values), "Rest"
                        else:
                            # Interpretar como nota, estimando como média das previsões não-rest.
                            h = round(
                                statistics.mean([
                                    12 * math.log2(freq / C0) - ideal_offset for freq in non_zero_values
                                ])
                            )
                            octave = h // 12
                            n = h % 12
                            if n < 0 or n >= len(note_names):
                                st.warning(f"Índice de nota inválido: {n}. Nota será ignorada.")
                                return float('inf'), "Rest"
                            note = note_names[n] + str(octave)
                            # Erro de quantização é a diferença total da nota quantizada.
                            error = sum([
                                abs(12 * math.log2(freq / C0) - ideal_offset - h)
                                for freq in non_zero_values
                            ])
                            return error, note

                    # Agrupar pitches em notas (simplificação: usar uma janela deslizante)
                    predictions_per_eighth = 40  # Aumentou de 20 para 40
                    prediction_start_offset = 0  # Ajustar conforme necessário

                    def get_quantization_and_error(pitch_outputs_and_rests, predictions_per_eighth,
                                                   prediction_start_offset, ideal_offset):
                        # Aplicar o offset inicial - podemos simplesmente adicionar o offset como rests.
                        pitch_outputs_and_rests = [0] * prediction_start_offset + list(pitch_outputs_and_rests)
                        # Coletar as previsões para cada nota (ou rest).
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

                    # Obter a melhor quantização
                    best_error = float("inf")
                    best_notes_and_rests = None
                    best_predictions_per_eighth = None

                    for ppn in range(15, 35, 1):
                        for pso in range(ppn):
                            error, notes_and_rests = get_quantization_and_error(
                                confident_pitch_values_hz, ppn,
                                pso, ideal_offset
                            )
                            if error < best_error:
                                best_error = error
                                best_notes_and_rests = notes_and_rests
                                best_predictions_per_eighth = ppn

                    # Remover rests iniciais e finais
                    while best_notes_and_rests and best_notes_and_rests[0] == 'Rest':
                        best_notes_and_rests = best_notes_and_rests[1:]
                    while best_notes_and_rests and best_notes_and_rests[-1] == 'Rest':
                        best_notes_and_rests = best_notes_and_rests[:-1]

                    # Garantir que todas as notas e rests são strings
                    best_notes_and_rests = [str(note) for note in best_notes_and_rests]

                    st.write(f"Notas e Rests Detectados: {best_notes_and_rests}")

                    # Criar a partitura musical usando music21
                    create_music_score(best_notes_and_rests, tempo, showScore, tmp_audio_path)

            except Exception as e:
                st.error(f"Erro ao processar o áudio: {e}")
            finally:
                # Remover arquivos temporários
                try:
                    os.remove(tmp_audio_path)
                except Exception as e:
                    st.warning(f"Erro ao remover arquivos temporários: {e}")

        st.header("Classificação de Novo Áudio")
        st.write("""
        **Envie um arquivo de áudio para ser classificado pelo modelo treinado.**
        
        **Explicação para Leigos:**
        - **O que você faz:** Envie um arquivo de áudio (como um canto, fala ou som ambiente) para que o modelo identifique a qual categoria ele pertence.
        - **O que acontece:** O aplicativo analisará o áudio, determinará a classe mais provável e mostrará a confiança dessa previsão. Além disso, você poderá visualizar a forma de onda, o espectrograma e as notas musicais detectadas.
        
        **Explicação para Técnicos:**
        - **Processo:** O áudio carregado é pré-processado para garantir a taxa de amostragem de 16kHz e convertido para mono. Em seguida, os embeddings são extraídos usando o modelo YAMNet. O classificador treinado em PyTorch utiliza esses embeddings para prever a classe do áudio, fornecendo uma pontuação de confiança baseada na função softmax.
        - **Detecção de Pitch:** Utilizando o modelo SPICE, o aplicativo realiza a detecção de pitch no áudio, convertendo os valores normalizados para Hz e quantizando-os em notas musicais utilizando a biblioteca `music21`. As notas detectadas são visualizadas e podem ser convertidas em um arquivo MIDI para reprodução.
        """)
        uploaded_audio = st.file_uploader("Faça upload do arquivo de áudio para classificação", type=["wav", "mp3", "ogg", "flac"])

        if uploaded_audio is not None:
            classify_new_audio(uploaded_audio)

    # Documentação e Agradecimentos
    st.write("### Documentação dos Procedimentos")
    st.write("""
    1. **Baixando e Preparando Arquivos de Áudio**: Você pode baixar arquivos de áudio de exemplo ou carregar seus próprios arquivos para começar.
    
    2. **Upload de Dados Supervisionados**: Envie um arquivo ZIP contendo subpastas, onde cada subpasta representa uma classe com seus respectivos arquivos de áudio.
    
    3. **Data Augmentation**: Se selecionado, aplica métodos de data augmentation como adição de ruído, estiramento de tempo e mudança de pitch nos dados de treinamento. Você pode ajustar os parâmetros `rate` e `n_steps` para controlar a intensidade dessas transformações.
    
    4. **Balanceamento de Classes**: Se selecionado, aplica métodos de balanceamento como oversampling (SMOTE) ou undersampling para tratar classes desbalanceadas.
    
    5. **Extração de Embeddings**: Utilizamos o YAMNet para extrair embeddings dos arquivos de áudio enviados.
    
    6. **Treinamento do Classificador**: Com os embeddings extraídos e após as opções de data augmentation e balanceamento, treinamos um classificador personalizado conforme os parâmetros definidos na barra lateral.
    
    7. **Classificação de Novo Áudio**: Após o treinamento, você pode enviar um novo arquivo de áudio para ser classificado pelo modelo treinado. O aplicativo exibirá a classe predita, a confiança, visualizará a forma de onda e o espectrograma do áudio carregado, além de realizar a detecção de pitch com SPICE para auxiliar na classificação.
    
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

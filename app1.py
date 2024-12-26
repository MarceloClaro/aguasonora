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
import requests
import math
import statistics
import music21
import streamlit.components.v1 as components
import pretty_midi
import soundfile as sf
import warnings

# Suprimir avisos relacionados ao torch.classes
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
    Se o sample rate não for 16 kHz, realiza o resample.
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
    Estica o tempo do áudio usando a função do Librosa.
    """
    return librosa.effects.time_stretch(waveform, rate)

def apply_pitch_shift(waveform, sr, n_steps=2):
    """
    Muda a altura do áudio (pitch shift).
    """
    return librosa.effects.pitch_shift(waveform, sr, n_steps)

def perform_data_augmentation(waveform, sr, augmentation_methods, rate=1.1, n_steps=2):
    """
    Aplica uma lista de métodos de data augmentation no áudio.
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
                shifted = apply_pitch_shift(waveform, sr, n_steps=n_steps)
                augmented_waveforms.append(shifted)
            except Exception as e:
                st.warning(f"Erro ao aplicar Pitch Shift: {e}")
    return augmented_waveforms

def extract_yamnet_embeddings(yamnet_model, audio_path):
    """
    Extrai embeddings usando o modelo YAMNet para um arquivo de áudio.
    Retorna a classe predita e a média dos embeddings (mean) das frames.
    """
    basename_audio = os.path.basename(audio_path)
    try:
        sr_orig, wav_data = wavfile.read(audio_path)
        st.write(f"Processando {basename_audio}: Sample Rate = {sr_orig}, Shape = {wav_data.shape}, Dtype = {wav_data.dtype}")
        
        # Converter para mono se for estéreo
        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)
            st.write(f"Convertido para mono: Shape = {wav_data.shape}")
        
        # Normalização do áudio para [-1, 1]
        if wav_data.dtype.kind == 'i':
            max_val = np.iinfo(wav_data.dtype).max
            waveform = wav_data / max_val
            st.write(f"Normalizado de inteiros para float: max_val = {max_val}")
        elif wav_data.dtype.kind == 'f':
            waveform = wav_data
            if np.max(waveform) > 1.0 or np.min(waveform) < -1.0:
                waveform = waveform / np.max(np.abs(waveform))
                st.write("Normalizado para o intervalo [-1.0, 1.0]")
        else:
            raise ValueError(f"Tipo de dado do áudio não suportado: {wav_data.dtype}")
    
        waveform = waveform.astype(np.float32)
    
        # Garantir sample_rate = 16000
        sr, waveform = ensure_sample_rate(sr_orig, waveform)
        st.write(f"Sample Rate ajustado: {sr}")
    
        # Executar o modelo YAMNet
        scores, embeddings, spectrogram = yamnet_model(waveform)
        st.write(f"Embeddings extraídos: Shape = {embeddings.shape}")
    
        # Extração da classe com maior score
        scores_np = scores.numpy()
        mean_scores = scores_np.mean(axis=0)
        pred_class = mean_scores.argmax()
        st.write(f"Classe predita pelo YAMNet: {pred_class}")
    
        # Cálculo da média dos embeddings
        mean_embedding = embeddings.numpy().mean(axis=0)
        st.write(f"Média dos embeddings das frames: Shape = {mean_embedding.shape}")
    
        return pred_class, mean_embedding
    except Exception as e:
        st.error(f"Erro ao processar {basename_audio}: {e}")
        return -1, None

def balance_classes(X, y, method):
    """
    Aplica balanceamento das classes (oversampling ou undersampling).
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

def train_audio_classifier(X_train, y_train, X_val, y_val,
                           input_dim, num_classes, epochs,
                           learning_rate, batch_size, l2_lambda, patience):
    """
    Treina um classificador simples em PyTorch com métricas de avaliação.
    """
    # Converter para tensores PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Modelo simples em PyTorch
    class AudioClassifier(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    best_val_loss = float('inf')
    best_model_wts = None
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    progress_bar = st.progress(0)

    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = classifier(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc.item())

        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)

        st.write(f"### Época {epoch+1}/{epochs}")
        st.write(f"**Treino - Perda:** {epoch_loss:.4f}, **Acurácia:** {epoch_acc.item():.4f}")
        st.write(f"**Validação - Perda:** {val_epoch_loss:.4f}, **Acurácia:** {val_epoch_acc.item():.4f}")

        # Early Stopping
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_wts = classifier.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                st.write("Early stopping!")
                break

    # Carregar os melhores pesos
    if best_model_wts is not None:
        classifier.load_state_dict(best_model_wts)

    # Plotar curvas de perda e acurácia
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(range(1, len(train_losses)+1), train_losses, label='Treino')
    ax[0].plot(range(1, len(val_losses)+1), val_losses, label='Validação')
    ax[0].set_title('Perda por Época')
    ax[0].set_xlabel('Épocas')
    ax[0].set_ylabel('Perda')
    ax[0].legend()

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
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # Curva ROC e AUC
    st.write("### Curva ROC")
    if num_classes > 1:
        from sklearn.preprocessing import label_binarize
        y_test_binarized = label_binarize(all_labels, classes=range(num_classes))
        y_pred_binarized = label_binarize(all_preds, classes=range(num_classes))

        if y_test_binarized.shape[1] > 1:
            fpr, tpr, roc_auc_dict = dict(), dict(), dict()
            for i in range(num_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
                roc_auc_dict[i] = auc(fpr[i], tpr[i])

            fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
            colors = sns.color_palette("hsv", num_classes)
            for i, color in zip(range(num_classes), colors):
                ax_roc.plot(fpr[i], tpr[i], color=color, lw=2,
                            label=f'Classe {i} (AUC = {roc_auc_dict[i]:0.2f})')

            ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('Taxa de Falsos Positivos')
            ax_roc.set_ylabel('Taxa de Verdadeiros Positivos')
            ax_roc.set_title('Curva ROC')
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
    Converte um arquivo MIDI para WAV usando um SoundFont (FluidR3_GM.sf2, por exemplo).
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        audio_data = midi_data.synthesize(fs=16000, sf2_path=soundfont_path)
        sf.write(wav_path, audio_data, 16000)
        return True
    except Exception as e:
        st.error(f"Erro ao converter MIDI para WAV: {e}")
        return False

def download_soundfont(soundfont_dir='sounds', soundfont_filename='FluidR3_GM.sf2'):
    """
    Verifica se o SoundFont existe. Se não, faz o download e salva na pasta especificada.
    """
    soundfont_path = os.path.join(soundfont_dir, soundfont_filename)
    if not os.path.exists(soundfont_path):
        st.info("Baixando SoundFont necessário para criar partitura...")
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
            st.markdown(f"**[Clique aqui para baixar manualmente o SoundFont `FluidR3_GM.sf2`](https://musical-artifacts.com/artifacts/738/FluidR3_GM.sf2/download)**")
            return None
    else:
        st.info("SoundFont já existe na pasta. Pulando o download.")
    return soundfont_path

def showScore(score):
    """
    Renderiza a partitura musical usando OpenSheetMusicDisplay via componente HTML do Streamlit.
    """
    xml = score.write('musicxml', fp=None)
    if not isinstance(xml, str):
        st.error("Falha ao converter a partitura para MusicXML como string.")
        return
    sanitized_xml = xml.replace('\\', '\\\\').replace('`', '\\`')
    showMusicXML(sanitized_xml)

def showMusicXML(xml):
    """
    Exibe o MusicXML usando OSMD (OpenSheetMusicDisplay).
    """
    DIV_ID = "OSMD_div"
    html_content = f"""
    <div id="{DIV_ID}">Carregando OpenSheetMusicDisplay...</div>
    <script>
    (function() {{
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
    Cria e renderiza a partitura musical usando music21,
    e converte MIDI para WAV com um SoundFont baixado automaticamente.
    """
    if not best_notes_and_rests:
        st.error("Nenhum dado de notas e rests fornecido para criar a partitura.")
        return

    # Criação da partitura com music21
    sc = music21.stream.Score()
    part = music21.stream.Part()
    sc.insert(0, part)

    # Assinatura de tempo (4/4)
    time_signature = music21.meter.TimeSignature('4/4')
    part.insert(0, time_signature)

    # Marcação de Tempo (BPM)
    a = music21.tempo.MetronomeMark(number=tempo)
    part.insert(1, a)

    # Inserir notas e rests
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

    # Verificar se a partitura está bem-formada
    if not sc.isWellFormedNotation():
        st.error("A partitura criada não está bem-formada. Verifique os dados de entrada.")
        return

    # Exibir a partitura usando OpenSheetMusicDisplay
    showScore(sc)

    # Gerar arquivo MIDI
    converted_audio_file_as_midi = tmp_audio_path[:-4] + '.mid'
    sc.write('midi', fp=converted_audio_file_as_midi)

    # Download ou verificação do SoundFont
    soundfont_path = download_soundfont()
    if soundfont_path is None:
        st.error("Falha ao obter o SoundFont. Não será possível converter MIDI para WAV.")
        return

    # Converter MIDI para WAV
    converted_audio_file_as_wav = tmp_audio_path[:-4] + '.wav'
    success = midi_to_wav(converted_audio_file_as_midi, converted_audio_file_as_wav, soundfont_path)

    if not success:
        st.error("Falha na conversão de MIDI para WAV.")
        return

    # Disponibilizar o WAV para download e reprodução
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

def main():
    # Configurações iniciais do Streamlit
    st.set_page_config(page_title="Classificação de Áudio com YAMNet e SPICE", layout="wide")
    st.title("Classificação de Áudio com YAMNet e Detecção de Pitch com SPICE")
    st.write("""
    Este aplicativo permite treinar um classificador de áudio supervisionado utilizando o modelo **YAMNet** para extrair embeddings e 
    o modelo **SPICE** para detecção de pitch, gerando uma partitura (MIDI/WAV).
    """)

    # Sidebar: Parâmetros
    st.sidebar.header("Configurações")

    st.sidebar.subheader("Parâmetros de Treinamento")
    epochs = st.sidebar.number_input("Épocas:", min_value=1, max_value=500, value=50, step=1)
    learning_rate = st.sidebar.select_slider("Taxa de Aprendizagem:", [0.1, 0.01, 0.001, 0.0001], 0.001)
    batch_size = st.sidebar.selectbox("Tamanho de Lote:", [8, 16, 32, 64], 1)
    l2_lambda = st.sidebar.number_input("Regularização L2:", 0.0, 0.1, 0.01, 0.01)
    patience = st.sidebar.number_input("Paciência (Early Stopping):", 1, 10, 3, 1)

    # Data Augmentation
    st.sidebar.subheader("Data Augmentation")
    augment = st.sidebar.checkbox("Aplicar Data Augmentation")
    if augment:
        augmentation_methods = st.sidebar.multiselect(
            "Escolha os métodos:",
            ["Add Noise", "Time Stretch", "Pitch Shift"],
            ["Add Noise", "Time Stretch"]
        )
        rate = st.sidebar.slider("Fator de Time Stretch:", 0.5, 2.0, 1.1, 0.1)
        n_steps = st.sidebar.slider("Semitons (Pitch Shift):", -12, 12, 2, 1)
    else:
        augmentation_methods = []
        rate, n_steps = 1.1, 2

    # Balanceamento de Classes
    st.sidebar.subheader("Balanceamento de Classes")
    balance_method = st.sidebar.selectbox("Método de Balanceamento:", ["None", "Oversample", "Undersample"], 0)

    # Secão principal: Download de Áudios Exemplo e Leitura
    st.header("Baixando e Preparando Áudios de Exemplo")
    sample_audio_1, sample_audio_2 = 'speech_whistling2.wav', 'miaow_16k.wav'
    sample_audio_1_url = "https://storage.googleapis.com/audioset/speech_whistling2.wav"
    sample_audio_2_url = "https://storage.googleapis.com/audioset/miaow_16k.wav"

    def download_audio(url, filename):
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(resp.content)
            st.success(f"Arquivo {filename} baixado com sucesso.")
        except Exception as e:
            st.error(f"Erro ao baixar {filename}: {e}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"Baixar {sample_audio_1}"):
            download_audio(sample_audio_1_url, sample_audio_1)
    with col2:
        if st.button(f"Baixar {sample_audio_2}"):
            download_audio(sample_audio_2_url, sample_audio_2)

    st.subheader("Tocar Arquivo de Áudio de Exemplo")
    sample_choice = st.selectbox("Selecione um áudio exemplo:", ["Nenhum", sample_audio_1, sample_audio_2])
    if sample_choice != "Nenhum" and os.path.exists(sample_choice):
        try:
            sr_sample, audio_data = wavfile.read(sample_choice)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            if sr_sample != 16000:
                sr_sample, audio_data = ensure_sample_rate(sr_sample, audio_data)
            if audio_data.dtype.kind == 'i':
                audio_data = audio_data / np.iinfo(audio_data.dtype).max
            elif audio_data.dtype.kind == 'f':
                if np.max(audio_data) > 1.0 or np.min(audio_data) < -1.0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
            audio_data = audio_data.astype(np.float32)
            dur = len(audio_data) / sr_sample
            st.write(f"Sample rate: {sr_sample}, Duração: {dur:.2f}s")
            st.audio(audio_data, format='audio/wav', sample_rate=sr_sample)
        except Exception as e:
            st.error(f"Erro ao reproduzir áudio: {e}")

    # Secão para Upload de Dados Supervisionados
    st.header("Upload de Dados Supervisionados (ZIP)")
    st.write("Estrutura esperada do ZIP: cada subpasta corresponde a uma classe, contendo os arquivos de áudio.")
    uploaded_zip = st.file_uploader("Envie um arquivo ZIP:", ["zip"])

    if uploaded_zip is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "uploaded.zip")
            with open(zip_path, 'wb') as f:
                f.write(uploaded_zip.read())
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                st.success("Arquivo ZIP extraído.")
            except zipfile.BadZipFile:
                st.error("Arquivo inválido! Por favor, envie um ZIP válido.")
                st.stop()

            classes_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
            if len(classes_dirs) < 2:
                st.error("Precisamos de pelo menos duas classes (subpastas).")
                st.stop()

            st.write(f"Classes encontradas: {classes_dirs}")
            yamnet_model = load_yamnet_model()

            embeddings, labels = [], []
            label_mapping = {cls_name: idx for idx, cls_name in enumerate(classes_dirs)}

            # Contagem total de arquivos
            total_files = 0
            for c in classes_dirs:
                cpath = os.path.join(tmpdir, c)
                total_files += len([f for f in os.listdir(cpath) if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))])

            processed_files = 0
            pbar = st.progress(0)

            for c in classes_dirs:
                cpath = os.path.join(tmpdir, c)
                audio_files = [os.path.join(cpath, f)
                               for f in os.listdir(cpath)
                               if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
                for afile in audio_files:
                    pred_cls, emb = extract_yamnet_embeddings(yamnet_model, afile)
                    if emb is not None:
                        if augment:
                            # Carregar audio com librosa
                            try:
                                wave, srx = librosa.load(afile, sr=16000, mono=True)
                                aug_waves = perform_data_augmentation(wave, srx, augmentation_methods, rate=rate, n_steps=n_steps)
                                for aw in aug_waves:
                                    temp_aug_path = os.path.join(tmpdir, "temp_aug.wav")
                                    sf.write(temp_aug_path, aw, srx)
                                    _, emb_aug = extract_yamnet_embeddings(yamnet_model, temp_aug_path)
                                    if emb_aug is not None:
                                        embeddings.append(emb_aug)
                                        labels.append(label_mapping[c])
                                    os.remove(temp_aug_path)
                            except Exception as e:
                                st.warning(f"Falha no Data Augmentation de {afile}: {e}")
                        else:
                            embeddings.append(emb)
                            labels.append(label_mapping[c])
                    processed_files += 1
                    pbar.progress(processed_files / total_files)

            if not embeddings:
                st.error("Falha ao extrair qualquer embedding. Verifique formatos de áudio.")
                st.stop()

            # Verificar se todos embeddings têm a mesma forma
            shapes = [e.shape for e in embeddings]
            if len(set(shapes)) != 1:
                st.error(f"Embeddings com formas diferentes: {set(shapes)}")
                st.stop()

            # Conversão para NumPy
            try:
                embeddings = np.array(embeddings)
                labels = np.array(labels)
                st.write(f"Embeddings: {embeddings.shape}")
            except ValueError as ve:
                st.error(f"Erro ao converter embeddings em array: {ve}")
                st.stop()

            # Balanceamento
            if balance_method != "None":
                embeddings_bal, labels_bal = balance_classes(embeddings, labels, balance_method)
            else:
                embeddings_bal, labels_bal = embeddings, labels

            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    embeddings_bal, labels_bal, test_size=0.2, random_state=42, stratify=labels_bal
                )
                st.write(f"Treino: {len(X_train)}, Validação: {len(X_val)}")
            except ValueError as v:
                st.error(f"Erro ao dividir dados: {v}")
                st.stop()

            # Treinar o modelo
            model = train_audio_classifier(X_train, y_train, X_val, y_val,
                                           input_dim=embeddings.shape[1],
                                           num_classes=len(classes_dirs),
                                           epochs=epochs,
                                           learning_rate=learning_rate,
                                           batch_size=batch_size,
                                           l2_lambda=l2_lambda,
                                           patience=patience)
            st.session_state['classifier'] = model
            st.session_state['classes'] = classes_dirs
            st.success("Treinamento concluído!")

            # Disponibilizar download do modelo
            buf = io.BytesIO()
            torch.save(model.state_dict(), buf)
            buf.seek(0)
            st.download_button(
                "Baixar Modelo Treinado",
                data=buf,
                file_name="modelo_treinado.pth",
                mime="application/octet-stream"
            )
            # Mapping
            class_map_txt = "\n".join(f"{k}:{v}" for k,v in label_mapping.items())
            st.download_button(
                "Baixar Classes",
                data=class_map_txt,
                file_name="class_mapping.txt",
                mime="text/plain"
            )

    # Classificação de Novo Áudio
    if 'classifier' in st.session_state and 'classes' in st.session_state:
        st.header("Classificação de Novo Áudio")
        st.write("""
        Envie um áudio para classificar e gerar a partitura musical.
        O aplicativo extrairá embeddings com YAMNet, classificará o áudio
        e exibirá a detecção de pitch (SPICE) com conversão para MIDI/WAV.
        """)

        uploaded_audio_file = st.file_uploader("Escolha o arquivo de áudio", type=["wav", "mp3", "ogg", "flac"])

        if uploaded_audio_file is not None:
            # Função interna
            def classify_new_audio(uploaded_audio):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmp:
                        tmp.write(uploaded_audio.read())
                        tmp_path = tmp.name
                    # Carregar e reproduzir
                    try:
                        srx, wdata = wavfile.read(tmp_path)
                        if wdata.ndim > 1:
                            wdata = wdata.mean(axis=1)
                        if srx != 16000:
                            srx, wdata = ensure_sample_rate(srx, wdata)
                        if wdata.dtype.kind == 'i':
                            wdata = wdata / np.iinfo(wdata.dtype).max
                        elif wdata.dtype.kind == 'f':
                            if np.max(wdata) > 1.0 or np.min(wdata) < -1.0:
                                wdata = wdata / np.max(np.abs(wdata))
                        wdata = wdata.astype(np.float32)
                        st.audio(wdata, format='audio/wav', sample_rate=srx)
                    except Exception as e:
                        st.warning(f"Não foi possível reproduzir o áudio: {e}")

                    # Extrair embeddings
                    yamnet_model = load_yamnet_model()
                    pc, emb = extract_yamnet_embeddings(yamnet_model, tmp_path)
                    if emb is None or pc == -1:
                        st.error("Falha ao extrair embeddings deste áudio.")
                        return

                    # Classificar
                    model = st.session_state['classifier']
                    classes_list = st.session_state['classes']

                    emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
                    model.eval()
                    with torch.no_grad():
                        out = model(emb_tensor)
                        probs = torch.nn.functional.softmax(out, dim=1)
                        conf, pred = torch.max(probs, 1)
                        class_idx = pred.item()
                        class_name = classes_list[class_idx]
                        st.write(f"**Classe Predita:** {class_name}")
                        st.write(f"**Confiança:** {conf.item():.4f}")

                    # Visualização do áudio (forma de onda e espectrograma)
                    fig, ax = plt.subplots(2, 1, figsize=(10,6))
                    ax[0].plot(wdata)
                    ax[0].set_title("Forma de Onda")
                    S = librosa.feature.melspectrogram(y=wdata, sr=srx, n_mels=128)
                    S_DB = librosa.power_to_db(S, ref=np.max)
                    librosa.display.specshow(S_DB, sr=srx, x_axis='time', y_axis='mel', ax=ax[1])
                    fig.colorbar(ax[1].collections[0], ax=ax[1], format='%+2.0f dB')
                    ax[1].set_title("Espectrograma (Mel)")
                    st.pyplot(fig)
                    plt.close(fig)

                    # Detecção de pitch com SPICE
                    spice_model = hub.load("https://tfhub.dev/google/spice/2")
                    wave_normalized = wdata / np.max(np.abs(wdata)) if np.max(np.abs(wdata)) != 0 else wdata

                    mo = spice_model.signatures["serving_default"](tf.constant(wave_normalized, tf.float32))
                    pitch_arr = mo["pitch"].numpy().flatten()
                    uncert_arr = mo["uncertainty"].numpy().flatten()
                    conf_arr = 1.0 - uncert_arr

                    fig2, ax2 = plt.subplots(figsize=(20, 6))
                    ax2.plot(pitch_arr, label='Pitch')
                    ax2.plot(conf_arr, label='Confiança')
                    ax2.legend()
                    ax2.set_title("Pitch e Confiança (SPICE)")
                    st.pyplot(fig2)
                    plt.close(fig2)

                    # Filtrar pitches confiáveis
                    good_idx = np.where(conf_arr >= 0.8)[0]
                    good_pitches = pitch_arr[good_idx]

                    # Converter para Hz
                    def output2hz(p):
                        PT_OFFSET = 25.58
                        PT_SLOPE = 63.07
                        FMIN = 10.0
                        BINS_PER_OCTAVE = 12.0
                        cqt_bin = p * PT_SLOPE + PT_OFFSET
                        return FMIN * 2.0 ** (1.0 * cqt_bin / BINS_PER_OCTAVE)

                    good_hz = [output2hz(g) for g in good_pitches]

                    # Plot pitches confiáveis
                    fig3, ax3 = plt.subplots(figsize=(20,6))
                    ax3.scatter(good_idx, good_hz, c="r", label='Pitch Confiante (Hz)')
                    ax3.set_ylim([0, 2000])
                    ax3.set_xlabel("Amostras")
                    ax3.set_ylabel("Hz")
                    ax3.set_title("Pitch Alto-Confiança (SPICE)")
                    ax3.legend()
                    st.pyplot(fig3)
                    plt.close(fig3)

                    # Converter pitches em notas
                    A4 = 440
                    C0 = A4 * pow(2, -4.75)
                    note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

                    def hz2offset(fr):
                        if fr == 0:
                            return None
                        try:
                            h = round(12 * math.log2(fr / C0))
                            return 12 * math.log2(fr / C0) - h
                        except ValueError:
                            return None

                    offsets = [hz2offset(p) for p in good_hz if p != 0 and hz2offset(p) is not None]
                    ideal_offset = float(statistics.mean(offsets)) if offsets else 0.0
                    st.write(f"**Offset Ideal:** {ideal_offset:.4f}")

                    def quantize(group, off):
                        nz = [v for v in group if v != 0]
                        zc = len(group) - len(nz)
                        if zc > 0.8 * len(group):
                            return 0.51 * len(nz), "Rest"
                        if not nz:
                            return float('inf'), "Rest"
                        try:
                            h = round(statistics.mean([12 * math.log2(v / C0) - off for v in nz]))
                        except ValueError:
                            return float('inf'), "Rest"
                        octv = h // 12
                        nn = h % 12
                        if nn < 0 or nn >= len(note_names):
                            return float('inf'), "Rest"
                        note_str = note_names[nn] + str(octv)
                        err = sum([
                            abs(12 * math.log2(v / C0) - off - h) for v in nz
                        ])
                        return err, note_str

                    # Agrupamento simplificado
                    step_size = 40
                    offset_init = 0
                    # Transformar good_hz em array de tamanhos step_size
                    splitted = [good_hz[i:i+step_size] for i in range(0, len(good_hz), step_size)]

                    best_error = float("inf")
                    best_notes = None
                    for ppn in range(15, 35, 1):
                        for pso in range(ppn):
                            # Montar
                            zero_pad = [0]*pso
                            merged = zero_pad + list(good_hz)
                            splitted2 = [merged[i:i+ppn] for i in range(0, len(merged), ppn)]
                            local_err = 0
                            local_notes = []
                            for sp in splitted2:
                                err_val, nt = quantize(sp, ideal_offset)
                                local_err += err_val
                                local_notes.append(nt)
                            if local_err < best_error:
                                best_error = local_err
                                best_notes = local_notes

                    # Remover rests no início/fim
                    while best_notes and best_notes[0] == 'Rest':
                        best_notes = best_notes[1:]
                    while best_notes and best_notes[-1] == 'Rest':
                        best_notes = best_notes[:-1]

                    best_notes = [str(b) for b in best_notes]
                    st.write("**Notas e Rests Detectados:**", best_notes)

                    # Estimativa de BPM
                    try:
                        tempo, _ = librosa.beat.beat_track(y=wdata, sr=srx)
                        if isinstance(tempo, np.ndarray):
                            if tempo.size == 1:
                                tempo = float(tempo.item())
                            else:
                                tempo = float(tempo[0])
                        elif not isinstance(tempo, (int, float)):
                            tempo = 120.0
                        st.write(f"**BPM Estimado:** {tempo:.2f}")
                    except Exception as e:
                        tempo = 120.0
                        st.warning(f"Falha ao estimar BPM: {e}")

                    # Criar partitura
                    create_music_score(best_notes, tempo, showScore, tmp_path)

                classify_new_audio(uploaded_audio_file)

    st.write("""
    ### Documentação e Agradecimentos
    - Este aplicativo foi ajustado para baixar automaticamente o SoundFont `FluidR3_GM.sf2`, facilitando a criação e
      conversão de partituras em MIDI/WAV.
    - Desenvolvido por Marcelo Claro.
    """)

if __name__ == "__main__":
    main()

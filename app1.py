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
    Ajusta a taxa de amostragem de um áudio para o valor desejado.
    """
    if original_sr != desired_sr:
        desired_length = int(round(float(len(waveform)) / original_sr * desired_sr))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sr, waveform

def extract_yamnet_embeddings(yamnet_model, audio_path):
    """
    Extrai embeddings usando o modelo YAMNet para um arquivo de áudio.
    Retorna a classe predita e a média dos embeddings das frames.
    """
    basename_audio = os.path.basename(audio_path)
    try:
        sr_orig, wav_data = wavfile.read(audio_path)

        # Verificar se está estéreo
        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)  # Converter para mono

        # Normalizar os dados do áudio
        if wav_data.dtype.kind == 'i':
            max_val = np.iinfo(wav_data.dtype).max
            waveform = wav_data / max_val
        elif wav_data.dtype.kind == 'f':
            waveform = wav_data
        else:
            raise ValueError(f"Tipo de dado do áudio não suportado: {wav_data.dtype}")

        # Garantir que o áudio esteja no formato float32 e ajuste de sample rate
        waveform = waveform.astype(np.float32)
        sr, waveform = ensure_sample_rate(sr_orig, waveform)

        # Executar o modelo YAMNet
        scores, embeddings, spectrogram = yamnet_model(waveform)
        scores_np = scores.numpy()
        mean_scores = scores_np.mean(axis=0)  # Média por frame
        pred_class = mean_scores.argmax()
        mean_embedding = embeddings.numpy().mean(axis=0)  # Embedding fixo

        return pred_class, mean_embedding
    except Exception as e:
        st.error(f"Erro ao processar {basename_audio}: {e}")
        return -1, None
def download_soundfont_uploader():
    """
    Permite que o usuário faça upload do SoundFont FluidR3_GM.sf2.
    Retorna o caminho para o SoundFont ou None se não for carregado.
    """
    st.sidebar.header("Configuração do SoundFont")
    st.sidebar.write("**Faça upload do arquivo SoundFont `FluidR3_GM.sf2` necessário para gerar a partitura.**")
    soundfont_file = st.sidebar.file_uploader("Upload do SoundFont (FluidR3_GM.sf2)", type=["sf2"])

    if soundfont_file is not None:
        soundfont_dir = 'sounds'
        os.makedirs(soundfont_dir, exist_ok=True)
        soundfont_path = os.path.join(soundfont_dir, "FluidR3_GM.sf2")
        with open(soundfont_path, "wb") as f:
            f.write(soundfont_file.read())
        st.sidebar.success("SoundFont carregado com sucesso.")
        return soundfont_path
    else:
        st.sidebar.warning("Por favor, faça upload do SoundFont `FluidR3_GM.sf2` para habilitar a criação de partituras.")
        return None

def create_music_score(best_notes_and_rests, tempo, showScore, tmp_audio_path, soundfont_path):
    """
    Cria e renderiza a partitura musical usando music21.
    """
    if best_notes_and_rests:
        sc = music21.stream.Score()
        part = music21.stream.Part()
        sc.insert(0, part)

        # Adicionar a assinatura de tempo (compasso 4/4)
        part.insert(0, music21.meter.TimeSignature('4/4'))
        part.insert(1, music21.tempo.MetronomeMark(number=tempo))

        # Adicionar notas e rests
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
            showScore(sc)

            # Salvar a partitura como arquivo MIDI
            converted_audio_file_as_midi = tmp_audio_path[:-4] + '.mid'
            sc.write('midi', fp=converted_audio_file_as_midi)

            # Converter MIDI para WAV
            converted_audio_file_as_wav = tmp_audio_path[:-4] + '.wav'
            success = midi_to_wav(converted_audio_file_as_midi, converted_audio_file_as_wav, soundfont_path)

            if success:
                # Oferecer o arquivo WAV para download e reprodução
                with open(converted_audio_file_as_wav, 'rb') as f:
                    wav_data = f.read()
                st.download_button(
                    label="Download do Arquivo WAV",
                    data=wav_data,
                    file_name=os.path.basename(converted_audio_file_as_wav),
                    mime="audio/wav"
                )
                st.audio(wav_data, format='audio/wav')
        else:
            st.error("A partitura criada não está bem-formada. Verifique os dados de entrada.")
    else:
        st.error("Nenhum dado de notas e rests fornecido para criar a partitura.")

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
def classify_new_audio(uploaded_audio, yamnet_model, soundfont_path):
    """
    Classifica um arquivo de áudio carregado e gera uma partitura com as notas detectadas.
    """
    try:
        # Salvar o arquivo de áudio temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmp_audio:
            tmp_audio.write(uploaded_audio.read())
            tmp_audio_path = tmp_audio.name

        # Extrair embeddings com YAMNet
        pred_class, embedding = extract_yamnet_embeddings(yamnet_model, tmp_audio_path)

        if embedding is not None and pred_class != -1:
            st.write(f"Classe Predita pelo YAMNet: {pred_class}")
            
            # Valores simulados para notas e BPM (pode ser substituído por lógica de detecção real)
            tempo = 120  # BPM padrão
            best_notes_and_rests = ["C4", "E4", "G4", "Rest", "F4", "A4", "Rest", "C5"]

            # Criar e exibir a partitura
            create_music_score(
                best_notes_and_rests, 
                tempo, 
                lambda score: st.write(score), 
                tmp_audio_path, 
                soundfont_path
            )
        else:
            st.error("Não foi possível classificar o áudio ou gerar uma partitura.")
    except Exception as e:
        st.error(f"Erro ao processar o áudio: {e}")
    finally:
        # Remover o arquivo temporário
        try:
            os.remove(tmp_audio_path)
        except Exception as e:
            st.warning(f"Erro ao remover arquivo temporário: {e}")
def main():
    st.set_page_config(page_title="Classificação de Áudio", layout="wide")
    st.title("Classificação de Áudio e Criação de Partituras")

    # Verificar e carregar o SoundFont
    soundfont_path = download_soundfont_uploader()

    # Upload do arquivo de áudio
    uploaded_audio = st.file_uploader("Faça upload de um arquivo de áudio para classificação", type=["wav", "mp3", "ogg", "flac"])

    if uploaded_audio:
        if soundfont_path:
            # Carregar o modelo YAMNet
            yamnet_model = load_yamnet_model()
            st.success("Modelo YAMNet carregado.")

            # Classificar o novo áudio
            classify_new_audio(uploaded_audio, yamnet_model, soundfont_path)
        else:
            st.warning("Por favor, faça upload do SoundFont `FluidR3_GM.sf2` na barra lateral para habilitar a criação de partituras.")
def plot_metrics(y_true, y_pred, class_names):
    """
    Gera relatórios de classificação, matriz de confusão e outras métricas visuais.
    """
    # Relatório de Classificação
    st.write("### Relatório de Classificação")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    st.write(pd.DataFrame(report).transpose())

    # Matriz de Confusão
    st.write("### Matriz de Confusão")
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # Curva ROC e AUC
    if len(class_names) > 2:
        st.write("### Curva ROC e AUC")
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        y_pred_bin = label_binarize(y_pred, classes=range(len(class_names)))
        
        fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
            auc_score = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc_score:.2f})')

        ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('Taxa de Falsos Positivos')
        ax_roc.set_ylabel('Taxa de Verdadeiros Positivos')
        ax_roc.set_title('Curva ROC')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    # F1-Score
    st.write("### F1-Score por Classe")
    f1_scores = f1_score(y_true, y_pred, average=None)
    f1_df = pd.DataFrame({'Classe': class_names, 'F1-Score': f1_scores})
    st.write(f1_df)
def main():
    st.set_page_config(page_title="Classificação de Áudio", layout="wide")
    st.title("Classificação de Áudio e Criação de Partituras")

    # Verificar e carregar o SoundFont
    soundfont_path = download_soundfont_uploader()

    # Upload do arquivo de áudio
    uploaded_audio = st.file_uploader("Faça upload de um arquivo de áudio para classificação", type=["wav", "mp3", "ogg", "flac"])

    if uploaded_audio:
        if soundfont_path:
            # Carregar o modelo YAMNet
            yamnet_model = load_yamnet_model()
            st.success("Modelo YAMNet carregado.")

            # Classificar o novo áudio
            classify_new_audio(uploaded_audio, yamnet_model, soundfont_path)

            # Simulação de classificação real para métricas (substitua pelos dados reais)
            y_true = [0, 1, 1, 2, 2]  # Classes verdadeiras simuladas
            y_pred = [0, 1, 2, 2, 2]  # Classes previstas simuladas
            class_names = ["Classe A", "Classe B", "Classe C"]

            # Exibir métricas
            plot_metrics(y_true, y_pred, class_names)
        else:
            st.warning("Por favor, faça upload do SoundFont `FluidR3_GM.sf2` na barra lateral para habilitar a criação de partituras.")
def classify_new_audio(uploaded_audio, yamnet_model, soundfont_path):
    """
    Processa e classifica o áudio carregado, gera partituras e resultados.
    """
    try:
        # Salvar o arquivo de áudio temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmp_audio:
            tmp_audio.write(uploaded_audio.read())
            tmp_audio_path = tmp_audio.name

        # Extrair embeddings e classificar
        pred_class, embedding = extract_yamnet_embeddings(yamnet_model, tmp_audio_path)

        if embedding is not None and pred_class != -1:
            st.write(f"Classe Predita pelo YAMNet: {pred_class}")

            # Simulação de geração de notas e tempos (substitua pelos valores reais)
            tempo = 120  # BPM padrão
            best_notes_and_rests = ["C4", "E4", "G4", "Rest", "F4", "A4"]  # Exemplo de notas

            # Criar e exibir a partitura
            create_music_score(
                best_notes_and_rests, tempo,
                showScore=lambda score: st.write(score),
                tmp_audio_path=tmp_audio_path,
                soundfont_path=soundfont_path
            )
        else:
            st.error("Erro ao processar o áudio: Não foi possível classificar ou extrair informações.")
    except Exception as e:
        st.error(f"Erro ao processar o áudio: {e}")
    finally:
        # Limpar arquivos temporários
        try:
            os.remove(tmp_audio_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()

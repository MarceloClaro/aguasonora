import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import streamlit as st
import tempfile
from PIL import Image
import os
import logging
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import shap
from pydub import AudioSegment
import statistics
import math
import librosa.display
from scipy.io import wavfile
import csv
import tensorflow as tf
import scipy.signal

# Configuração de Logging
logging.basicConfig(
    filename='experiment_logs.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Função para definir sementes (seed) para reprodutibilidade
def set_seeds(seed):
    """Define as sementes para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Função para carregar arquivos de áudio
def carregar_audio(caminho_arquivo, sr=None):
    """Carrega o arquivo de áudio usando Librosa."""
    try:
        data, sr = librosa.load(caminho_arquivo, sr=sr, res_type='kaiser_fast')
        return data, sr
    except Exception as e:
        logging.error(f"Erro ao carregar áudio: {e}")
        return None, None

# Função para extrair features do áudio
def extrair_features(data, sr, use_mfcc=True, use_spectral_centroid=True):
    """
    Extrai MFCCs e centróide espectral do áudio. Normaliza as features.
    """
    try:
        features_list = []
        if use_mfcc:
            mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            features_list.append(mfccs_scaled)
        if use_spectral_centroid:
            centroid = librosa.feature.spectral_centroid(y=data, sr=sr)
            centroid_mean = np.mean(centroid, axis=1)
            features_list.append(centroid_mean)
        if len(features_list) > 1:
            features_vector = np.concatenate(features_list, axis=0)
        else:
            features_vector = features_list[0]
        # Normalização
        features_vector = (features_vector - np.mean(features_vector)) / (np.std(features_vector) + 1e-9)
        return features_vector
    except Exception as e:
        logging.error(f"Erro ao extrair features: {e}")
        return None

# Função para aplicar augmentations (aumento de dados)
def aumentar_audio(data, sr, augmentations):
    """Aplica augmentations no áudio."""
    try:
        return augmentations(samples=data, sample_rate=sr)
    except Exception as e:
        logging.error(f"Erro ao aumentar áudio: {e}")
        return data

# Função para carregar o modelo YAMNet
def carregar_modelo_yamnet():
    """Carrega o modelo YAMNet do TensorFlow Hub."""
    model_url = 'https://tfhub.dev/google/yamnet/1'
    model = hub.load(model_url)
    return model

# Função para carregar os nomes das classes do YAMNet
def class_names_from_csv(class_map_csv_text):
    """Retorna a lista de nomes das classes a partir do CSV de mapeamento."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

# Função para verificar e ajustar a taxa de amostragem
def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

# Função para preparar o arquivo de áudio
def preparar_audio(wav_file_name):
    """Prepara o arquivo de áudio, garantindo a taxa de amostragem de 16kHz."""
    sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
    # Normalize the audio to [-1.0, 1.0]
    waveform = wav_data / tf.int16.max
    return waveform, sample_rate

# Função para prever usando YAMNet
def classificar_audio_yamnet(waveform, model, class_names):
    """Classifica o áudio utilizando o modelo YAMNet."""
    scores, embeddings, spectrogram = model(waveform)
    scores_np = scores.numpy()
    infered_class = class_names[scores_np.mean(axis=0).argmax()]
    return infered_class, scores_np

# Função para exibir os resultados
def exibir_resultados(infered_class, scores_np, waveform, spectrogram):
    """Exibe os resultados da classificação."""
    plt.figure(figsize=(10, 6))
    # Plot the waveform.
    plt.subplot(3, 1, 1)
    plt.plot(waveform)
    plt.xlim([0, len(waveform)])

    # Plot the log-mel spectrogram (returned by the model).
    plt.subplot(3, 1, 2)
    plt.imshow(spectrogram.T, aspect='auto', interpolation='nearest', origin='lower')

    # Plot and label the model output scores for the top-scoring classes.
    mean_scores = np.mean(scores_np, axis=0)
    top_n = 10
    top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
    plt.subplot(3, 1, 3)
    plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')

    patch_padding = (0.025 / 2) / 0.01
    plt.xlim([-patch_padding-0.5, scores_np.shape[0] + patch_padding-0.5])
    yticks = range(0, top_n, 1)
    plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
    plt.ylim(-0.5 + np.array([top_n, 0]))
    plt.show()

# Função principal para carregamento e classificação
def processar_audio_classificacao():
    model = carregar_modelo_yamnet()  # Carrega o modelo YAMNet
    class_map_path = model.class_map_path().numpy()
    class_names = class_names_from_csv(class_map_path)  # Carrega os nomes das classes
    
    audio_file = st.file_uploader("Upload do Áudio para Classificação", type=["wav", "mp3", "flac", "ogg", "m4a"])

    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_audio:
            tmp_audio.write(audio_file.read())
            caminho_audio = tmp_audio.name

        waveform, sample_rate = preparar_audio(caminho_audio)  # Prepara o áudio
        infered_class, scores_np = classificar_audio_yamnet(waveform, model, class_names)  # Classifica o áudio
        st.markdown(f"**Classe Predita:** {infered_class}")
        exibir_resultados(infered_class, scores_np, waveform, spectrogram)  # Exibe os resultados

# Definir e controlar o SEED de reprodução
seed_options = list(range(0, 61, 2))
default_seed = 42
SEED = default_seed
set_seeds(SEED)

# Interface do Streamlit
st.sidebar.header("Configurações Gerais")
app_mode = st.sidebar.radio("Escolha a seção", ["Classificar Áudio com YAMNet", "Treinar Modelo"])

if app_mode == "Classificar Áudio com YAMNet":
    processar_audio_classificacao()
elif app_mode == "Treinar Modelo":
    treinar_modelo(SEED)

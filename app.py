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
import zipfile

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

# Função para visualizar representações do áudio
def visualizar_audio(data, sr):
    """Visualiza diferentes representações do áudio."""
    try:
        # Forma de onda
        fig_wave, ax_wave = plt.subplots(figsize=(8,4))
        librosa.display.waveshow(data, sr=sr, ax=ax_wave)
        ax_wave.set_title("Forma de Onda no Tempo")
        ax_wave.set_xlabel("Tempo (s)")
        ax_wave.set_ylabel("Amplitude")
        st.pyplot(fig_wave)
        plt.close(fig_wave)

        # FFT (Espectro)
        fft = np.fft.fft(data)
        fft_abs = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(data), 1/sr)[:len(fft)//2]
        fig_fft, ax_fft = plt.subplots(figsize=(8,4))
        ax_fft.plot(freqs, fft_abs)
        ax_fft.set_title("Espectro (Amplitude x Frequência)")
        ax_fft.set_xlabel("Frequência (Hz)")
        ax_fft.set_ylabel("Amplitude")
        st.pyplot(fig_fft)
        plt.close(fig_fft)

        # Espectrograma
        D = np.abs(librosa.stft(data))**2
        S = librosa.power_to_db(D, ref=np.max)
        fig_spec, ax_spec = plt.subplots(figsize=(8,4))
        img_spec = librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='hz', ax=ax_spec)
        ax_spec.set_title("Espectrograma")
        fig_spec.colorbar(img_spec, ax=ax_spec, format='%+2.0f dB')
        st.pyplot(fig_spec)
        plt.close(fig_spec)

        # MFCCs Plot
        mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
        fig_mfcc, ax_mfcc = plt.subplots(figsize=(8,4))
        img_mfcc = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax_mfcc)
        ax_mfcc.set_title("MFCCs")
        fig_mfcc.colorbar(img_mfcc, ax=ax_mfcc)
        st.pyplot(fig_mfcc)
        plt.close(fig_mfcc)
    except Exception as e:
        st.error(f"Erro na visualização do áudio: {e}")
        logging.error(f"Erro na visualização do áudio: {e}")

# Função para treinar o modelo CNN
def treinar_modelo(SEED):
    with st.expander("Treinamento do Modelo CNN"):
        st.markdown("### Passo 1: Upload do Dataset (ZIP)")
        zip_upload = st.file_uploader("Upload do ZIP", type=["zip"])

        if zip_upload is not None:
            try:
                st.write("Extraindo o Dataset...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                    tmp_zip.write(zip_upload.read())
                    caminho_zip = tmp_zip.name

                # Extração do dataset
                diretorio_extracao = tempfile.mkdtemp()
                with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
                    zip_ref.extractall(diretorio_extracao)
                caminho_base = diretorio_extracao

                # Leitura das classes e arquivos
                categorias = [d for d in os.listdir(caminho_base) if os.path.isdir(os.path.join(caminho_base, d))]
                if len(categorias) == 0:
                    st.error("Nenhuma subpasta encontrada no ZIP.")
                    return

                st.success("Dataset extraído!")
                st.write(f"Classes encontradas: {', '.join(categorias)}")

                caminhos_arquivos = []
                labels = []
                for cat in categorias:
                    caminho_cat = os.path.join(caminho_base, cat)
                    arquivos_na_cat = [f for f in os.listdir(caminho_cat) if f.lower().endswith(('.wav','.mp3','.flac','.ogg','.m4a'))]
                    st.write(f"Classe '{cat}': {len(arquivos_na_cat)} arquivos.")
                    for nome_arquivo in arquivos_na_cat:
                        caminhos_arquivos.append(os.path.join(caminho_cat, nome_arquivo))
                        labels.append(cat)

                df = pd.DataFrame({'caminho_arquivo': caminhos_arquivos, 'classe': labels})
                st.write("10 Primeiras Amostras do Dataset:")
                st.dataframe(df.head(10))

                if len(df) == 0:
                    st.error("Nenhuma amostra encontrada no dataset.")
                    return

                # Codificação das classes
                labelencoder = LabelEncoder()
                y = labelencoder.fit_transform(df['classe'])
                classes = labelencoder.classes_

                st.write(f"Classes codificadas: {', '.join(classes)}")

                # Extração de Features (MFCCs, Centróide)
                st.write("Extraindo Features (MFCCs, Centróide)...")
                X = []
                y_valid = []
                for i, row in df.iterrows():
                    arquivo = row['caminho_arquivo']
                    data, sr = carregar_audio(arquivo, sr=None)
                    if data is not None:
                        ftrs = extrair_features(data, sr, use_mfcc=True, use_spectral_centroid=True)
                        if ftrs is not None:
                            X.append(ftrs)
                            y_valid.append(y[i])

                X = np.array(X)
                y_valid = np.array(y_valid)
                st.write(f"Features extraídas: {X.shape}")

                # Ajuste do Modelo com TensorFlow
                modelo = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(X.shape[1],)),
                    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(len(classes), activation='softmax')
                ])

                modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                # Treinamento do Modelo
                st.write("Treinando Modelo...")
                modelo.fit(X, y_valid, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

                # Avaliação e Métricas
                st.write("Modelo Treinado!")
                st.write("Evaluando o Modelo...")
                loss, acc = modelo.evaluate(X, y_valid, verbose=0)
                st.write(f"Loss: {loss}")
                st.write(f"Acurácia: {acc}")

                # Salvar o modelo
                modelo.save('modelo_classificador_agua.h5')
                st.download_button("Baixar Modelo", 'modelo_classificador_agua.h5', file_name="modelo_classificador_agua.h5")

                st.success("Treinamento Concluído!")

            except Exception as e:
                st.error(f"Erro: {e}")
                logging.error(f"Erro: {e}")

# Função para classificação de novos áudios com YAMNet
def classificar_audio_yamnet(waveform, model):
    """Classifica o áudio utilizando o modelo YAMNet."""
    scores, embeddings, spectrogram = model(waveform)
    scores_np = scores.numpy()
    
    # Carrega as classes a partir do modelo
    class_names = class_names_from_csv(model.class_map_path().numpy())

    # Encontra a classe com a maior pontuação média
    infered_class = class_names[scores_np.mean(axis=0).argmax()]
    return infered_class, scores_np

# Função para exibir os resultados
def exibir_resultados(infered_class, scores_np, waveform, spectrogram):
    """Exibe os resultados do modelo."""
    st.markdown(f"**Classe Predita:** {infered_class}")

    # Plot do espectrograma
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(waveform)
    plt.xlim([0, len(waveform)])
    
    # Plot do log-mel spectrogram (retornado pelo modelo)
    plt.subplot(3, 1, 2)
    plt.imshow(spectrogram.numpy().T, aspect='auto', interpolation='nearest', origin='lower')
    
    # Plot das pontuações de classe para as classes de maior pontuação
    mean_scores = np.mean(scores_np, axis=0)
    top_n = 10
    top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
    
    plt.subplot(3, 1, 3)
    plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')
    plt.xlim([-0.5, scores_np.shape[0]-0.5])
    
    # Rotula as classes mais pontuadas
    yticks = range(0, top_n, 1)
    plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
    plt.ylim(-0.5 + np.array([top_n, 0]))
    st.pyplot()

# Função para carregar o áudio e preparar para classificação
def preparar_audio(caminho_arquivo):
    sample_rate, waveform = wavfile.read(caminho_arquivo)
    waveform, _ = ensure_sample_rate(sample_rate, waveform)
    return waveform, sample_rate

# Função para garantir que a taxa de amostragem é 16kHz
def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

# Interface do Streamlit
st.sidebar.header("Configurações Gerais")
app_mode = st.sidebar.radio("Escolha a seção", ["Classificar Áudio com YAMNet", "Treinar Modelo"])

if app_mode == "Classificar Áudio com YAMNet":
    processar_audio_classificacao()
elif app_mode == "Treinar Modelo":
    treinar_modelo(SEED)

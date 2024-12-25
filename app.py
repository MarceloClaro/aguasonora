import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import scipy.signal
from scipy.io import wavfile
import csv
import matplotlib.pyplot as plt
import librosa.display
from io import BytesIO

# ---------------
# Ajuste de Layout e Config
# ---------------
st.set_page_config(page_title="Classificação de Áudio (YAMNet)", layout="centered")

# ---------------
# Cache: Carregar o modelo YAMNet
# ---------------
@st.cache_data(show_spinner=True)
def carregar_modelo_yamnet():
    """
    Carrega o modelo YAMNet do TensorFlow Hub.
    """
    model_url = "https://tfhub.dev/google/yamnet/1"
    yamnet_model = hub.load(model_url)
    return yamnet_model

# ---------------
# Cache: Carregar as classes a partir do CSV do modelo
# ---------------
@st.cache_data(show_spinner=True)
def carregar_classes(_yamnet_model):
    """
    Carrega o arquivo CSV de classes (class_map) do modelo YAMNet.
    Uso de _yamnet_model como nome de parâmetro para evitar hashing do objeto de modelo.
    """
    # Caminho do CSV de classes
    class_map_path = _yamnet_model.class_map_path().numpy().decode("utf-8")

    class_names = []
    with tf.io.gfile.GFile(class_map_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row["display_name"])
    return class_names

# ---------------
# Funções Auxiliares
# ---------------
def ensure_sample_rate(original_sr, waveform, desired_sr=16000):
    """
    Garante que o áudio esteja em desired_sr (16kHz).
    Caso o original_sr seja diferente, faz o resample usando scipy.signal.resample.
    """
    if original_sr != desired_sr:
        desired_length = int(round(len(waveform) / original_sr * desired_sr))
        waveform = scipy.signal.resample(waveform, desired_length)
        return desired_sr, waveform
    else:
        return original_sr, waveform

def normalizar_waveform(wav_data):
    """
    Converte wav_data (int16 ou float) em float32 normalizado (-1.0..1.0).
    """
    # Se for int16, normaliza por 32767 (ou 32768). Caso já seja float, apenas converte
    if wav_data.dtype == np.int16:
        wav_data = wav_data.astype(np.float32) / 32768.0
    elif wav_data.dtype == np.float32 or wav_data.dtype == np.float64:
        wav_data = wav_data.astype(np.float32)
    else:
        wav_data = wav_data.astype(np.float32)  # fallback
    return wav_data

def inferir_classe(waveform, yamnet_model, class_names):
    """
    Executa inferência no waveform usando YAMNet.
    Retorna a classe majoritária e o array de scores (frames x 521).
    """
    # O modelo retorna (scores, embeddings, log_mel_spectrogram)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    scores_np = scores.numpy()  # shape: (n_frames, 521)
    spectrogram_np = spectrogram.numpy()

    # Classe inferida (média ao longo dos frames, pegando argmax)
    mean_scores = scores_np.mean(axis=0)
    infered_class_idx = np.argmax(mean_scores)
    infered_class = class_names[infered_class_idx]

    return infered_class, scores_np, spectrogram_np


# ---------------
# Aplicativo Streamlit
# ---------------
def main():
    st.title("Classificação de Áudio com YAMNet")

    st.markdown("""
    Este aplicativo faz classificação de sons usando o modelo [YAMNet](https://tfhub.dev/google/yamnet/1), 
    que foi treinado em 521 classes de áudio (AudioSet).
    """)

    # Menu lateral
    st.sidebar.header("Configurações")
    st.sidebar.write("Upload de áudio para classificação:")
    audio_file = st.sidebar.file_uploader("Selecione um arquivo de áudio (WAV/MP3/OGG/etc.)", type=["wav","mp3","ogg","flac","m4a"])

    # Carregar o modelo (cacheado)
    yamnet_model = carregar_modelo_yamnet()
    # Carregar nomes das classes (cacheado)
    class_names = carregar_classes(yamnet_model)

    # Se um arquivo foi enviado:
    if audio_file is not None:
        # Ler o arquivo (usando scipy.io.wavfile ou librosa):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp.flush()
            sample_rate, wav_data = wavfile.read(tmp.name)

        # Ajustar sample rate
        sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data, 16000)
        # Normalizar
        waveform = normalizar_waveform(wav_data)

        # Precisamos de um tensor float32 [n_samples] para o modelo YAMNet
        tensor_waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)

        # Executar inferência
        infered_class, scores_np, spectrogram_np = inferir_classe(tensor_waveform, yamnet_model, class_names)

        # Mostrar resultado
        st.subheader(f"Classe Predita: {infered_class}")

        # Mostra player de áudio (16kHz)
        st.audio(audio_file, format="audio/wav")
        
        # Visualizações
        st.subheader("Visualizações")

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))
        fig.tight_layout(pad=3)

        # 1) Forma de onda
        axes[0].set_title("Forma de Onda")
        axes[0].plot(waveform)
        axes[0].set_xlim([0, len(waveform)])
        axes[0].set_xlabel("Amostras")
        axes[0].set_ylabel("Amplitude")

        # 2) Espectrograma log-mel retornado pelo modelo
        axes[1].set_title("Log-Mel Spectrogram (YAMNet)")
        im1 = axes[1].imshow(spectrogram_np.T, aspect='auto', origin='lower')
        fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.02)
        axes[1].set_ylabel("Mel bins")

        # 3) Scores (top classes)
        mean_scores = scores_np.mean(axis=0)
        top_N = 10
        top_index = np.argsort(mean_scores)[::-1][:top_N]
        axes[2].set_title(f"Scores (Top {top_N} classes)")
        show_scores = scores_np[:, top_index].T  # shape: (top_N, n_frames)
        im2 = axes[2].imshow(show_scores, aspect='auto', interpolation='nearest', cmap='gray_r')
        fig.colorbar(im2, ax=axes[2], orientation='vertical', fraction=0.02)
        axes[2].set_yticks(range(top_N))
        axes[2].set_yticklabels([class_names[i] for i in top_index])
        axes[2].set_xlabel("Frames")

        st.pyplot(fig)


if __name__ == "__main__":
    main()

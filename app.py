import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import scipy
import os
import zipfile
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ---------------------------------------------
# 1) Use cache_resource para objetos grandes (modelo TF Hub), evitando serialização.
# ---------------------------------------------
@st.cache_resource
def load_yamnet_model():
    """
    Carrega o modelo YAMNet do TF Hub (sem serialização).
    Retorna o objeto do modelo carregado.
    """
    yam_model = hub.load('https://tfhub.dev/google/yamnet/1')
    return yam_model

def ensure_sample_rate(original_sr, waveform, desired_sr=16000):
    """
    Resample se não estiver na taxa de amostragem 16 kHz.
    """
    if original_sr != desired_sr:
        desired_length = int(round(float(len(waveform)) / original_sr * desired_sr))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sr, waveform

def main():
    st.title("Classificação de Áudio com YAMNet - Exemplo Ajustado")
    st.write("""
    Este aplicativo exemplifica como carregar o modelo YAMNet via TF Hub sem
    gerar erro de serialização no Streamlit.
    """)

    # Upload de arquivo ZIP com áudios
    zip_file = st.file_uploader("Envie um arquivo .zip com áudios (.wav) e subpastas", type=["zip"])
    if zip_file is not None:
        # 1) Carrega o modelo usando cache_resource
        st.write("Carregando o modelo YAMNet do TF Hub...")
        yamnet_model = load_yamnet_model()
        st.success("Modelo YAMNet carregado com sucesso!")

        # 2) Extrair o ZIP em um diretório temporário
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "audios.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_file.read())
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir)

            # 3) Percorrer arquivos .wav e realizar predições
            audio_files = []
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if file.lower().endswith(".wav"):
                        audio_files.append(os.path.join(root, file))

            if len(audio_files) == 0:
                st.warning("Nenhum arquivo .wav encontrado no ZIP.")
                return

            st.write(f"Arquivos encontrados: {len(audio_files)}")
            predictions = []
            file_names = []
            for audio_path in audio_files:
                file_names.append(os.path.basename(audio_path))

                try:
                    sr, wav_data = wavfile.read(audio_path)
                    # Normaliza para [-1, 1]
                    waveform = wav_data / np.iinfo(wav_data.dtype).max
                    # Ajustar sample rate para 16k
                    sr, waveform = ensure_sample_rate(sr, waveform)

                    # 4) Rodar o modelo YAMNet
                    # yamnet_model retorna: scores, embeddings, spectrogram
                    scores, embeddings, log_mel = yamnet_model(waveform)

                    scores_np = scores.numpy()
                    mean_scores = scores_np.mean(axis=0)
                    top_class = mean_scores.argmax()
                    predictions.append(top_class)
                except Exception as e:
                    st.error(f"Erro ao processar o arquivo {audio_path}: {e}")
                    predictions.append(-1)

            # 5) Visualizar resultados
            st.subheader("Resultados")
            results_df = []
            for fn, pred in zip(file_names, predictions):
                results_df.append({"file": fn, "class_idx": pred})
            st.write(results_df)

            # 6) Exemplo de eventuais labels custom
            # Caso tenha um mapa de classes, substitua idx -> nome
            # (yAMNet possui 521 classes, mas se você usa outro subset, ajuste)
            # st.write(class_map[pred])

    else:
        st.info("Por favor, envie um arquivo .zip com áudios .wav")

if __name__ == "__main__":
    main()

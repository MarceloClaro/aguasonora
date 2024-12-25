import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal
import tempfile
import os

# Configuração de Logging (opcional)
import logging
logging.basicConfig(
    filename='yamnet_streamlit.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Função para carregar as classes a partir do arquivo CSV
def class_names_from_csv(class_map_csv_path):
    """Retorna uma lista de nomes de classes correspondentes ao vetor de pontuação."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

# Função para garantir a taxa de amostragem correta
def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """Reamostra o waveform se necessário."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) /
                                   original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

# Função para processar o áudio
def process_audio(file):
    try:
        # Salvar o arquivo de áudio temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(file.read())
            tmp_audio_path = tmp_audio.name

        # Ler o arquivo de áudio
        sample_rate, wav_data = wavfile.read(tmp_audio_path, 'rb')
        sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

        # Normalizar os dados para [-1.0, 1.0]
        if wav_data.dtype == np.int16:
            waveform = wav_data / 32768.0
        elif wav_data.dtype == np.int32:
            waveform = wav_data / 2147483648.0
        elif wav_data.dtype == np.float32 or wav_data.dtype == np.float64:
            waveform = wav_data
        else:
            st.error("Formato de áudio não suportado.")
            return None, None

        # Remover arquivos temporários
        os.remove(tmp_audio_path)

        return sample_rate, waveform
    except Exception as e:
        st.error(f"Erro ao processar o áudio: {e}")
        logging.error(f"Erro ao processar o áudio: {e}")
        return None, None

# Função para plotar a forma de onda
def plot_waveform(waveform, sample_rate):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(waveform)
    ax.set_title("Forma de Onda")
    ax.set_xlabel("Amostras")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)
    plt.close(fig)

# Função para plotar o espectrograma
def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(spectrogram.T, aspect='auto', interpolation='nearest', origin='lower')
    ax.set_title("Espectrograma (Log-Mel)")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Frequência (Hz)")
    st.pyplot(fig)
    plt.close(fig)

# Função para plotar as pontuações das classes
def plot_class_scores(scores, class_names, top_n=10):
    mean_scores = np.mean(scores, axis=0)
    top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
    top_class_scores = mean_scores[top_class_indices]
    top_class_names = [class_names[i] for i in top_class_indices]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_class_names[::-1], top_class_scores[::-1], color='skyblue')
    ax.set_xlabel("Pontuação Média")
    ax.set_title(f"Top {top_n} Classes Preditas")
    st.pyplot(fig)
    plt.close(fig)

# Carregar o modelo YAMNet do TensorFlow Hub
@st.cache_resource
def load_yamnet_model():
    try:
        model = hub.load('https://tfhub.dev/google/yamnet/1')
        logging.info("Modelo YAMNet carregado com sucesso.")
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo YAMNet: {e}")
        logging.error(f"Erro ao carregar o modelo YAMNet: {e}")
        return None

# Carregar as classes
@st.cache_data
def load_class_names(model):
    try:
        class_map_path = model.class_map_path().numpy().decode('utf-8')
        class_names = class_names_from_csv(class_map_path)
        logging.info("Classes carregadas com sucesso.")
        return class_names
    except Exception as e:
        st.error(f"Erro ao carregar as classes: {e}")
        logging.error(f"Erro ao carregar as classes: {e}")
        return []

# Função principal do Streamlit
def main():
    st.title("Classificação de Sons com YAMNet")
    st.markdown("""
    Este aplicativo permite que você faça upload de arquivos de áudio e classifique os sons utilizando o modelo **YAMNet** do TensorFlow Hub.
    """)
    
    # Carregar o modelo YAMNet
    model = load_yamnet_model()
    if model is None:
        st.stop()
    
    # Carregar as classes
    class_names = load_class_names(model)
    if not class_names:
        st.stop()
    
    # Upload do arquivo de áudio
    st.header("Upload do Arquivo de Áudio")
    audio_file = st.file_uploader("Faça upload de um arquivo de áudio (.wav)", type=["wav"])
    
    if audio_file is not None:
        sample_rate, waveform = process_audio(audio_file)
        if waveform is not None:
            st.subheader("Visualização do Áudio")
            plot_waveform(waveform, sample_rate)
            
            # Escutar o áudio
            st.audio(waveform, format='audio/wav', sample_rate=sample_rate)
            
            # Executar o modelo YAMNet
            st.subheader("Classificação com YAMNet")
            try:
                # Executar o modelo
                scores, embeddings, spectrogram = model(waveform)
                scores_np = scores.numpy()
                spectrogram_np = spectrogram.numpy()
                
                # Inferência da classe
                infered_class = class_names[scores_np.mean(axis=0).argmax()]
                st.write(f"**A principal classe inferida é:** {infered_class}")
                
                # Plotar o espectrograma
                st.subheader("Espectrograma")
                plot_spectrogram(spectrogram_np)
                
                # Plotar as pontuações das classes
                st.subheader("Top Classes Preditas")
                plot_class_scores(scores_np, class_names, top_n=10)
                
            except Exception as e:
                st.error(f"Erro ao executar o modelo YAMNet: {e}")
                logging.error(f"Erro ao executar o modelo YAMNet: {e}")
    else:
        st.info("Aguardando upload de um arquivo de áudio...")

    # Seção de Contexto e Descrição Completa
    with st.expander("Contexto e Descrição Completa"):
        st.markdown("""
        **Classificação de Sons com YAMNet**

        Este aplicativo utiliza o **YAMNet**, uma rede neural profunda pré-treinada que prevê 521 eventos de áudio a partir do corpus AudioSet-YouTube. YAMNet emprega a arquitetura **MobileNetV1** com convoluções separáveis por profundidade, o que a torna eficiente e eficaz para reconhecimento de eventos de áudio.

        **Como Funciona:**

        1. **Upload do Áudio:**  
           Faça upload de um arquivo de áudio no formato `.wav`.

        2. **Processamento do Áudio:**  
           O áudio é reamostrado para 16kHz se necessário e normalizado para valores em [-1.0, 1.0].

        3. **Classificação com YAMNet:**  
           O modelo YAMNet processa o áudio e retorna pontuações para 521 classes de áudio.

        4. **Visualização dos Resultados:**  
           - **Forma de Onda:** Visualização temporal do áudio.
           - **Espectrograma:** Representação espectral do áudio.
           - **Top Classes Preditas:** As classes com maior pontuação média.

        **Explicação para Leigos:**

        Imagine o YAMNet como um especialista em reconhecer diferentes sons. Você fornece um áudio, e ele identifica quais sons estão presentes, como vozes de animais, sons de veículos, música, etc. As visualizações ajudam a entender como o áudio se comporta no tempo e na frequência, facilitando a interpretação dos resultados.

        **Benefícios do YAMNet:**

        - **Pré-Treinado:** Economiza tempo e recursos, pois já foi treinado em um grande conjunto de dados.
        - **Versátil:** Capaz de reconhecer uma ampla variedade de eventos de áudio.
        - **Eficiente:** Utiliza arquiteturas modernas que equilibram precisão e desempenho.

        **Nota:**  
        Para melhores resultados, utilize arquivos de áudio claros, preferencialmente em formato `.wav` mono com taxa de amostragem de 16kHz.

        **Recursos Adicionais:**

        - [Modelo YAMNet no TensorFlow Hub](https://tfhub.dev/google/yamnet/1)
        - [Documentação do YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
        """)

if __name__ == "__main__":
    main()

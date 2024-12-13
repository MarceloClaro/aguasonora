import streamlit as st
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display as ld
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# Configuração de sementes para reproducibilidade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Função para carregar e processar áudio
def load_audio(file_path, sr=None):
    data, sr = librosa.load(file_path, sr=sr, res_type='kaiser_fast')
    return data, sr

# Função para extrair características MFCC
def extract_features(data, sr):
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Função para aumentar os dados de áudio
def augment_audio(data, sr):
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ])
    augmented_data = augment(samples=data, sample_rate=sr)
    return augmented_data

# Função para preparar os dados
def prepare_data(uploaded_files):
    file_paths = []
    labels = []
    for file in uploaded_files:
        file_path = os.path.join("temp", file.name)
        file_paths.append(file_path)
        # Suponha que as classes estão no nome do arquivo ou em uma estrutura de pastas
        labels.append("classe_placeholder")
    return file_paths, labels

# Função para treinar o modelo
def train_model(X, y):
    # Definir a arquitetura da CNN
    model = Sequential([
        Conv1D(64, kernel_size=10, activation='relu', input_shape=(X.shape[1], 1)),
        Dropout(0.4),
        MaxPooling1D(pool_size=4),
        Conv1D(128, kernel_size=10, activation='relu', padding='same'),
        Dropout(0.4),
        MaxPooling1D(pool_size=4),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(y.shape[1], activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Treinar o modelo
    history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1)
    return model

# Configuração da aplicação Streamlit
st.title('Classificação de Sons de Água em Copo de Vidro com CNN')

# Sidebar para entradas do usuário
st.sidebar.header('Configurações')
uploaded_files = st.sidebar.file_uploader("Carregue arquivos WAV para treinamento", accept_multiple_files=True)
train_button = st.sidebar.button('Treinar Modelo')

# Conteúdo principal
st.header('Carregamento e Pré-processamento de Dados')
if uploaded_files:
    st.write(f'Carregados {len(uploaded_files)} arquivos.')
    # Salvar arquivos temporariamente
    os.makedirs("temp", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("temp", file.name), "wb") as f:
            f.write(file.getvalue())
    # Preparar dados
    file_paths, labels = prepare_data(uploaded_files)
    # Extrair características
    X = []
    for file_path in file_paths:
        data, sr = load_audio(file_path, sr=None)
        features = extract_features(data, sr)
        X.append(features)
    X = np.array(X)
    y = to_categorical(labels)  # Ajustar de acordo com as classes reais
else:
    st.write('Nenhum arquivo carregado. Por favor, carregue arquivos WAV.')

# Seção de treinamento do modelo
if train_button:
    st.header('Treinamento do Modelo')
    if 'X' in locals() and 'y' in locals():
        with st.spinner('Treinando o modelo...'):
            model = train_model(X, y)
            st.session_state['model'] = model
        st.success('Modelo treinado com sucesso!')
    else:
        st.warning('Carregue os dados e tente novamente.')

# Seção de avaliação do modelo
st.header('Avaliação do Modelo')
if 'model' in st.session_state:
    # Avaliar o modelo
    score = st.session_state['model'].evaluate(X, y)
    st.write(f'Acurácia: {score[1]*100:.2f}%')
else:
    st.write('Treine o modelo para visualizar as métricas.')

# Seção de classificação de novo áudio
st.header('Classificação de Novo Áudio')
uploaded_audio = st.file_uploader("Carregue um arquivo WAV para classificar", type=['wav'])
if uploaded_audio:
    st.audio(uploaded_audio, format='audio/wav')
    classify_button = st.button('Classificar Áudio')
    if classify_button:
        if 'model' in st.session_state:
            # Processar o áudio carregado
            data, sr = load_audio(uploaded_audio.name, sr=None)
            features = extract_features(data, sr)
            features = np.expand_dims(features, axis=0)
            features = np.expand_dims(features, axis=2)
            # Fazer previsão
            prediction = st.session_state['model'].predict(features)
            predicted_class = np.argmax(prediction)
            st.write(f'Classe prevista: {predicted_class}')
        else:
            st.warning('Treine o modelo antes de realizar a classificação.')

# Seção de visualizações
st.header('Visualizações')
if uploaded_audio:
    # Plotar waveform
    data, sr = load_audio(uploaded_audio.name, sr=None)
    fig, ax = plt.subplots()
    ld.waveshow(data, sr=sr, ax=ax)
    ax.set_title('Waveform')
    st.pyplot(fig)
    
    # Plotar espectro de frequências
    fft = np.fft.fft(data)
    fft = np.abs(fft[:len(data)//2])
    freqs = np.fft.fftfreq(len(data), 1/sr)[:len(data)//2]
    fig, ax = plt.subplots()
    ax.plot(freqs, fft)
    ax.set_title('Espectro de Frequências')
    st.pyplot(fig)
    
    # Plotar espectrograma
    fig, ax = plt.subplots()
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max), ax=ax)
    ax.set_title('Spectrograma (STFT)')
    st.pyplot(fig)

# Rodapé
st.markdown('---')
st.write('Desenvolvido por [Seu Nome]')

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display as ld
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import streamlit as st
import tempfile
from PIL import Image, UnidentifiedImageError
import torch
import zipfile
import gc

# ==================== CONFIGURAÇÃO DA PÁGINA ====================
seed_options = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]
seed_selection = 42  # Valor padrão

# Configuração do SEED
seed_selection = st.sidebar.selectbox(
    "Escolha o valor do SEED:",
    options=seed_options,
    index=seed_options.index(42),
    help="Define a semente para reprodutibilidade dos resultados."
)
SEED = seed_selection  # Definindo a variável SEED

# Configuração do Logotipo e Imagem
capa_path = 'capa.png'
logo_path = "logo.png"

# ==================== FUNÇÕES DE PROCESSAMENTO ====================
augment_default = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
])

def carregar_audio(caminho_arquivo, sr=None):
    try:
        data, sr = librosa.load(caminho_arquivo, sr=sr, res_type='kaiser_fast')
        return data, sr
    except Exception as e:
        st.error(f"Erro ao carregar o áudio {caminho_arquivo}: {e}")
        return None, None

def extrair_features(data, sr):
    try:
        mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        st.error(f"Erro ao extrair MFCC: {e}")
        return None

def aumentar_audio(data, sr, augmentations):
    try:
        augmented_data = augmentations(samples=data, sample_rate=sr)
        return augmented_data
    except Exception as e:
        st.error(f"Erro ao aplicar Data Augmentation: {e}")
        return data

def plot_forma_onda(data, sr, titulo="Forma de Onda"):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(titulo, fontsize=16)
    ld.waveshow(data, sr=sr, ax=ax)
    ax.set_xlabel("Tempo (segundos)", fontsize=14)
    ax.set_ylabel("Amplitude", fontsize=14)
    st.pyplot(fig)
    plt.close(fig)

def plot_mfcc(data, sr, titulo="Espectrograma (MFCC)"):
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    mfccs_db = librosa.amplitude_to_db(np.abs(mfccs))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(titulo, fontsize=16)
    mappable = ld.specshow(mfccs_db, x_axis='time', y_axis='mel', cmap='Spectral', sr=sr, ax=ax)
    cbar = plt.colorbar(mappable=mappable, ax=ax, format='%+2.f dB')
    cbar.ax.set_ylabel("Intensidade (dB)", fontsize=14)
    ax.set_xlabel("Tempo (segundos)", fontsize=14)
    ax.set_ylabel("Frequência Mel", fontsize=14)
    st.pyplot(fig)
    plt.close(fig)

def plot_probabilidades_classes(class_probs, titulo="Probabilidades das Classes"):
    classes = list(class_probs.keys())
    probs = list(class_probs.values())
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=classes, y=probs, palette='viridis', ax=ax)
    ax.set_title(titulo, fontsize=16)
    ax.set_xlabel("Classes", fontsize=14)
    ax.set_ylabel("Probabilidade", fontsize=14)
    ax.set_ylim(0, 1)
    for i, v in enumerate(probs):
        ax.text(i, v + 0.01, f"{v*100:.2f}%", ha='center', va='bottom', fontsize=12)
    st.pyplot(fig)
    plt.close(fig)

def processar_novo_audio(caminho_audio, modelo, labelencoder):
    data, sr = carregar_audio(caminho_audio, sr=None)
    if data is None:
        return None, None, None

    mfccs = extrair_features(data, sr)
    if mfccs is None:
        return None, None, None

    mfccs = mfccs.reshape(1, -1, 1)

    prediction = modelo.predict(mfccs)
    pred_class = np.argmax(prediction, axis=1)
    pred_label = labelencoder.inverse_transform(pred_class)
    confidence = prediction[0][pred_class][0]
    class_probs = {labelencoder.classes_[i]: float(prediction[0][i]) for i in range(len(labelencoder.classes_))}
    return pred_label[0], confidence, class_probs

# ==================== FUNÇÕES DE TREINAMENTO ====================
def treinar_modelo(SEED):
    st.header("Treinamento do Modelo CNN")
    st.write("""
    ### Passo 1: Upload do Dataset
    O **dataset** deve estar organizado em um arquivo ZIP com pastas para cada classe. Por exemplo:
    dataset.zip/
    ├── agua_gelada/
    │   ├── arquivo1.wav
    │   └── ...
    └── ...
    """)

    zip_upload = st.file_uploader("Faça upload do arquivo ZIP contendo as pastas das classes", type=["zip"], key="dataset_upload")
    if zip_upload is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            tmp_zip.write(zip_upload.read())
            caminho_zip = tmp_zip.name

        # Extrair dados do ZIP
        diretorio_extracao = tempfile.mkdtemp()
        with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
            zip_ref.extractall(diretorio_extracao)
        caminho_base = diretorio_extracao
        categorias = [d for d in os.listdir(caminho_base) if os.path.isdir(os.path.join(caminho_base, d))]

        # Organize os arquivos de áudio
        caminhos_arquivos = []
        labels = []
        for cat in categorias:
            caminho_cat = os.path.join(caminho_base, cat)
            arquivos_na_cat = [f for f in os.listdir(caminho_cat) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
            for nome_arquivo in arquivos_na_cat:
                caminho_completo = os.path.join(caminho_cat, nome_arquivo)
                caminhos_arquivos.append(caminho_completo)
                labels.append(cat)

        df = pd.DataFrame({'caminho_arquivo': caminhos_arquivos, 'classe': labels})
        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(df['classe'])

        # Extração de características (MFCC)
        X = []
        for i, row in df.iterrows():
            arquivo = row['caminho_arquivo']
            data, sr = carregar_audio(arquivo, sr=None)
            if data is not None:
                features = extrair_features(data, sr)
                if features is not None:
                    X.append(features)

        X = np.array(X)
        y = np.array(y)

        # Divisão dos dados
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
        
        # Definição da CNN
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1], 1)),
            tf.keras.layers.Conv1D(64, kernel_size=10, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.MaxPooling1D(pool_size=4),
            tf.keras.layers.Conv1D(128, kernel_size=10, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.MaxPooling1D(pool_size=4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(len(labelencoder.classes_), activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Callbacks para salvar o melhor modelo e parar o treinamento cedo
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss', verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
        
        # Treinamento
        model.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train, epochs=50, batch_size=32,
                  validation_data=(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)), y_test), callbacks=callbacks)

        # Avaliação e gráficos
        st.write("### Avaliação do Modelo")
        st.write(f"Acurácia no conjunto de teste: {model.evaluate(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)), y_test)[1]*100:.2f}%")

        # Salvar modelo treinado
        model.save("final_model.h5")
        st.download_button(label="Baixar Modelo Treinado", data=open("final_model.h5", 'rb'), file_name="modelo_final.h5")
        
# ==================== INTERFACE STREAMLIT ====================
def classificar_audio(SEED):
    st.header("Classificação de Novo Áudio")
    model_file = st.file_uploader("Carregar Modelo Treinado", type=["h5", "keras", "pth"])

    if model_file is not None:
        # Carregar o modelo
        if model_file.name.endswith((".h5", ".keras")):
            model = load_model(model_file)
        else:
            st.error("Formato de modelo não suportado. Use .h5 ou .keras.")
            return

        classes_file = st.file_uploader("Carregar Arquivo de Classes (classes.txt)", type=["txt"])
        if classes_file is not None:
            classes = classes_file.read().decode("utf-8").splitlines()
            labelencoder = LabelEncoder()
            labelencoder.fit(classes)

            audio_upload = st.file_uploader("Carregar Áudio para Classificação", type=["wav", "mp3", "flac"])
            if audio_upload is not None:
                # Processar o áudio
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_upload.name)[1]) as tmp_audio:
                    tmp_audio.write(audio_upload.read())
                    caminho_audio = tmp_audio.name

                data, sr = carregar_audio(caminho_audio)
                if data is not None:
                    st.audio(caminho_audio)

                    # Predição
                    pred_label, confianca, class_probs = processar_novo_audio(caminho_audio, model, labelencoder)

                    if pred_label is not None:
                        st.write(f"Classe Predita: {pred_label}")
                        st.write(f"Confiança: {confianca*100:.2f}%")
                        plot_probabilidades_classes(class_probs)
                        plot_forma_onda(data, sr)
                        plot_mfcc(data, sr)

if __name__ == "__main__":
    app_mode = st.sidebar.selectbox("Escolha a Seção", ["Classificar Áudio", "Treinar Modelo"])
    if app_mode == "Classificar Áudio":
        classificar_audio(SEED)
    elif app_mode == "Treinar Modelo":
        treinar_modelo(SEED)

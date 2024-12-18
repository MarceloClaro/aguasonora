# ============================ Configurações Iniciais ============================
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, KFold
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import streamlit as st
import tempfile
from PIL import Image
import io
import torch
import zipfile
import gc
import os
import logging
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import shap
import torchvision.transforms as torch_transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset

# Configuração de logging
logging.basicConfig(
    filename='experiment_logs.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ============================ Definições de Funções Utilitárias ============================

def set_seeds(seed: int):
    """
    Define as sementes para reprodutibilidade.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================ Funções Auxiliares de Áudio ============================

def carregar_audio(caminho_arquivo: str, sr: int = None):
    """
    Carrega um arquivo de áudio usando Librosa.
    """
    if not os.path.exists(caminho_arquivo):
        logging.error(f"Arquivo de áudio não encontrado: {caminho_arquivo}")
        return None, None
    try:
        data, sr = librosa.load(caminho_arquivo, sr=sr, res_type='kaiser_fast', backend='soundfile')
        if len(data) == 0:
            logging.error(f"Arquivo de áudio vazio: {caminho_arquivo}")
            return None, None
        return data, sr
    except Exception as e:
        logging.error(f"Erro ao carregar áudio {caminho_arquivo}: {e}")
        return None, None

def extrair_features(data: np.ndarray, sr: int, use_mfcc=True, use_spectral_centroid=True):
    """
    Extrai features do áudio (MFCCs e Centróide Espectral).
    """
    try:
        features = []
        if use_mfcc:
            mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
            features.append(np.mean(mfccs.T, axis=0))
        if use_spectral_centroid:
            centroid = librosa.feature.spectral_centroid(y=data, sr=sr)
            features.append(np.mean(centroid, axis=1))
        if len(features) > 1:
            features_vector = np.concatenate(features, axis=0)
        else:
            features_vector = features[0]
        # Normalização
        return (features_vector - np.mean(features_vector)) / (np.std(features_vector) + 1e-9)
    except Exception as e:
        logging.error(f"Erro ao extrair features: {e}")
        return None

# ============================ Modularização Principal ============================

def criar_dataset_audio(df: pd.DataFrame, classes: list, transform=None):
    """
    Criação de dataset personalizado.
    """
    return AudioSpectrogramDataset(df, classes, transform)

def configurar_callbacks(save_path: str, monitor='val_loss', patience=5, mode='min'):
    """
    Configura callbacks para Keras.
    """
    os.makedirs(save_path, exist_ok=True)
    return [
        ModelCheckpoint(
            filepath=os.path.join(save_path, 'modelo_melhor.keras'),
            monitor=monitor,
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

# ============================ Treinamento e Avaliação ============================

def treinar_cnn_personalizada(X_train, y_train, X_val, y_val, params, callbacks, class_weights=None):
    """
    Treina uma CNN personalizada com os parâmetros especificados.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input

    modelo = Sequential()
    modelo.add(Input(shape=(X_train.shape[1], 1)))
    for filtros, kernel in zip(params['conv_filters'], params['conv_kernel_size']):
        modelo.add(Conv1D(filters=filtros, kernel_size=kernel, activation='relu', kernel_regularizer=params['regularization']))
        modelo.add(Dropout(params['dropout_rate']))
        modelo.add(MaxPooling1D(pool_size=2))
    modelo.add(Flatten())
    for unidades in params['dense_units']:
        modelo.add(Dense(units=unidades, activation='relu', kernel_regularizer=params['regularization']))
        modelo.add(Dropout(params['dropout_rate']))
    modelo.add(Dense(len(params['classes']), activation='softmax'))
    modelo.compile(
        loss='categorical_crossentropy',
        optimizer=params['optimizer'],
        metrics=['accuracy']
    )
    historico = modelo.fit(
        X_train, to_categorical(y_train),
        validation_data=(X_val, to_categorical(y_val)),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        callbacks=callbacks,
        class_weight=class_weights
    )
    return modelo, historico

# ============================ Interface de Streamlit ============================

def main_app():
    """
    Aplicação principal no Streamlit.
    """
    st.title("Classificador de Sons de Água")
    app_mode = st.sidebar.selectbox("Modo", ["Classificar Áudio", "Treinar Modelo"])
    SEED = st.sidebar.slider("SEED", 0, 100, 42)
    set_seeds(SEED)
    if app_mode == "Classificar Áudio":
        st.write("Funcionalidade em construção.")
    elif app_mode == "Treinar Modelo":
        st.write("Funcionalidade de treinamento.")

# ============================ Execução ============================
if __name__ == "__main__":
    main_app()

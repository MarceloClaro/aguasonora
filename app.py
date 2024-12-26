import os
import zipfile
import random
import tempfile
import uuid
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn, optim
import tensorflow as tf
import tensorflow_hub as hub
import scipy
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import streamlit as st
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
#                         Funções Auxiliares
# -----------------------------------------------------------------------------
@st.cache_data
def load_yamnet_model():
    """
    Carrega o modelo YAMNet do TensorFlow Hub.
    """
    return hub.load('https://tfhub.dev/google/yamnet/1')

@st.cache_data
def load_class_map(_model):
    """
    Carrega o CSV de classes do YAMNet (classe, índice, nome) e retorna apenas a coluna de nomes.
    """
    import csv
    class_map_path = _model.class_map_path().numpy().decode('utf-8')
    class_names = []
    with tf.io.gfile.GFile(class_map_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

def ensure_sample_rate(original_sr, waveform, desired_sr=16000):
    """
    Se o áudio não estiver em 16 kHz, faz resample com scipy.signal.resample.
    """
    if original_sr != desired_sr:
        desired_len = int(round(float(len(waveform)) / original_sr * desired_sr))
        waveform = scipy.signal.resample(waveform, desired_len)
    return desired_sr, waveform

def plot_confusion_matrix(y_true, y_pred, run_id, unique_labels):
    """
    Plota a matriz de confusão normalizada e disponibiliza download.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(unique_labels)), normalize='true')
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', xticklabels=unique_labels, yticklabels=unique_labels, ax=ax)
    ax.set_title('Matriz de Confusão (Normalizada)')
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    plt.tight_layout()

    filename = f'cm_run_{run_id}.png'
    plt.savefig(filename)
    st.image(filename, caption="Matriz de Confusão")
    with open(filename, 'rb') as f:
        st.download_button(
            label="Download da Matriz de Confusão",
            data=f,
            file_name=filename,
            mime="image/png"
        )
    plt.close()

def evaluate_on_test(model, X_test, y_test, unique_labels, run_id):
    """
    Aplica o modelo no conjunto de teste, gera predições e exibe métricas (relatório + matriz de confusão).
    Ajuste principal:
      - Usamos `labels=range(len(unique_labels))` no classification_report
    """
    st.write("Avaliando no conjunto de teste...")

    # Fazer predições
    pred = model(X_test)
    pred_classes = tf.math.argmax(pred, axis=1).numpy()

    # Plotar matriz de confusão
    plot_confusion_matrix(y_test, pred_classes, run_id, unique_labels)

    # Relatório de Classificação 
    # Forçando scikit-learn a considerar TODAS as classes definidas em `unique_labels`
    try:
        report = classification_report(
            y_test,
            pred_classes,
            target_names=unique_labels,
            labels=range(len(unique_labels)),  # <-- ajuste para evitar erro se y_test tiver só 1 classe
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.write("**Relatório de Classificação**:")
        st.dataframe(report_df)

        # Download do relatório
        report_file = f"classification_report_run_{run_id}.csv"
        report_df.to_csv(report_file, index=True)
        with open(report_file, "rb") as f:
            st.download_button(
                label="Download do Relatório de Classificação",
                data=f,
                file_name=report_file,
                mime="text/csv"
            )
    except ValueError as e:
        st.error(f"Erro ao gerar classification_report: {e}")
        logging.error(f"Erro ao gerar classification_report: {e}")

    # Curva ROC/AUC (para caso binário, por exemplo)
    if len(unique_labels) == 2:
        y_probs = tf.nn.softmax(pred, axis=1).numpy()[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs, pos_label=1)
        auc_val = roc_auc_score(y_test, y_probs)
        st.write(f"AUC: {auc_val:.4f}")

        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.legend()
        ax_roc.set_title("Curva ROC")

        roc_file = f'roc_run_{run_id}.png'
        fig_roc.savefig(roc_file)
        st.image(roc_file, caption="Curva ROC")
        with open(roc_file, 'rb') as f:
            st.download_button(
                label="Download Curva ROC",
                data=f,
                file_name=roc_file,
                mime="image/png"
            )
        plt.close(fig_roc)

def plot_loss_accuracy(history, run_id):
    """
    Dado o history de treinamento do Keras, plota e disponibiliza download
    dos gráficos de perda e acurácia por época.
    """
    if not history.history:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.history['loss'], label='Treino')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Validação')
    axes[0].set_title('Perda por Época')
    axes[0].set_xlabel('Épocas')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Accuracy
    if 'accuracy' in history.history:
        axes[1].plot(history.history['accuracy'], label='Treino')
    if 'val_accuracy' in history.history:
        axes[1].plot(history.history['val_accuracy'], label='Validação')
    axes[1].set_title('Acurácia por Época')
    axes[1].set_xlabel('Épocas')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    filename = f"loss_acc_run_{run_id}.png"
    fig.savefig(filename)
    st.image(filename, caption="Treino - Loss & Accuracy")
    with open(filename, 'rb') as f:
        st.download_button("Download Gráfico de Loss e Acurácia", data=f, file_name=filename, mime="image/png")
    plt.close(fig)

def train_audio_classification(yam_model, class_map, X_data, y_data,
                               train_split=0.7, valid_split=0.15,
                               epochs=10, learning_rate=0.001,
                               batch_size=4, run_id="0"):
    """
    Simples pipeline de treinamento com Keras para classificar com base nos embeddings do YAMNet.
    - X_data: array de forma (N, 1024) se já for embedding, ou algo similar
    - y_data: classes (int)
    * Ajuste para lidar com unbalanced test set (classification_report + labels=...)
    """

    unique_labels = list(sorted(set(y_data)))
    st.write(f"Labels encontrados (train+test): {unique_labels}")

    # Dividir dataset
    X_train, X_tmp, y_train, y_tmp = train_test_split(X_data, y_data,
                                                      test_size=(1 - train_split),
                                                      stratify=y_data,
                                                      random_state=42)
    valid_rel = valid_split / (1 - train_split) if (1 - train_split) > 0 else 0.5
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(1 - valid_rel),
        stratify=y_tmp, random_state=42
    )

    st.write(f"Tamanho Treino = {len(X_train)}, Validação = {len(X_valid)}, Teste = {len(X_test)}")

    # Modelo Keras simples
    input_dim = X_train.shape[1]  # ex: 1024 do embedding
    model_ff = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(unique_labels), activation='softmax')
    ])
    model_ff.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

    # Treinar
    st.write("Treinando rede fully-connected...")
    history = model_ff.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Plotar loss/acc
    plot_loss_accuracy(history, run_id)

    # Avaliar no teste
    evaluate_on_test(model_ff, X_test, y_test, unique_labels, run_id)

    return model_ff

def process_zip_and_extract_embeddings(zip_file, yam_model, class_map):
    """
    Exemplo: Carrega o ZIP, faz resample em 16k, extrai embeddings e devolve (X, y).
    A pipeline aqui é simplificada; cada pasta no ZIP = 1 classe etc.
    """
    import io
    import soundfile as sf

    # Descompactar num diretório temporário
    tdir = tempfile.mkdtemp()
    zpath = os.path.join(tdir, "audios.zip")
    with open(zpath, 'wb') as f:
        f.write(zip_file.read())
    with zipfile.ZipFile(zpath, 'r') as zip_ref:
        zip_ref.extractall(tdir)

    # Procurar pastas e arquivos
    # Supondo que cada subpasta = 1 classe
    data = []
    labels = []
    current_label = 0
    label_map = {}

    for root, dirs, files in os.walk(tdir):
        # Se 'files' contiver áudios, root é a pasta da classe
        if files:
            # Supondo root = something/nomedaclasse
            class_name = os.path.basename(root)
            if class_name not in label_map:
                label_map[class_name] = current_label
                current_label += 1
            for f_ in files:
                if f_.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                    audio_path = os.path.join(root, f_)
                    # Carregar áudio
                    try:
                        sr_orig, wav_data = sf.read(audio_path)
                        if len(wav_data.shape) > 1:
                            wav_data = np.mean(wav_data, axis=1)  # Convert para mono
                        sr, wav_data = ensure_sample_rate(sr_orig, wav_data, 16000)

                        # Normalizar
                        waveform = wav_data / 32768.0 if wav_data.dtype == np.int16 else wav_data

                        # Passar no YAMNet
                        scores, embeddings, spectrogram = yam_model(waveform)
                        embeddings_np = embeddings.numpy()
                        # Pegar a média do embedding
                        emb_mean = embeddings_np.mean(axis=0)
                        data.append(emb_mean)
                        labels.append(label_map[class_name])
                    except Exception as e:
                        logging.warning(f"Falha ao processar {audio_path}: {e}")

    data = np.array(data)
    labels = np.array(labels)
    # Mapeamento
    sorted_items = sorted(label_map.items(), key=lambda x: x[1])
    sorted_labels = [k for k, v in sorted_items]
    st.write(f"label_map = {label_map}")
    st.write(f"sorted_labels = {sorted_labels}")
    return data, labels, sorted_labels

# -----------------------------------------------------------------------------
#                               MAIN
# -----------------------------------------------------------------------------
def main():
    st.title("Classificação de Áudio com YAMNet + Rede Fully-Connected")
    zip_file = st.file_uploader("Envie um arquivo ZIP contendo subpastas de áudio", type=["zip"])
    epochs = st.sidebar.slider("Épocas:", 1, 50, 10)
    learning_rate = st.sidebar.select_slider("Learning Rate:", options=[0.1, 0.01, 0.001, 0.0001], value=0.001)
    batch_size = st.sidebar.selectbox("Batch Size:", options=[2, 4, 8, 16], index=1)
    train_split = st.sidebar.slider("Train (%)", 0.5, 0.9, 0.7)
    valid_split = st.sidebar.slider("Valid (%)", 0.05, 0.4, 0.15)

    if zip_file is not None:
        # Carrega YAMNet
        st.write("Carregando modelo YAMNet do TF Hub...")
        yam_model = load_yamnet_model()
        st.write("Carregando classe do YAMNet (class_map)...")
        class_map = load_class_map(yam_model)
        st.write(f"YAMNet OK. Tamanho do class_map={len(class_map)} (não necessariamente usado diretamente).")

        # Extrair embeddings
        st.write("Extraindo embeddings de áudio (via YAMNet)...")
        X_data, y_data, labels = process_zip_and_extract_embeddings(zip_file, yam_model, class_map)
        st.write(f"shape X_data={X_data.shape}, shape y_data={y_data.shape}, classes={labels}")

        if X_data.shape[0] < 2:
            st.error("Menos de 2 áudios processados. Impossível treinar.")
            return

        run_id = str(uuid.uuid4())[:8]
        st.write(f"Iniciando Treino - Run={run_id}")
        model_ff = train_audio_classification(
            yam_model, class_map, X_data, y_data,
            train_split=train_split, valid_split=valid_split,
            epochs=epochs, learning_rate=learning_rate,
            batch_size=batch_size, run_id=run_id
        )
        st.success("Treinamento concluído!")
    else:
        st.info("Por favor, faça upload de um arquivo ZIP contendo subpastas de áudio (por classe).")


if __name__ == "__main__":
    main()

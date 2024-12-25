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
import streamlit as st
import logging
import base64
import io
import warnings
import librosa
import librosa.display
import tensorflow as tf
import tensorflow_hub as hub
import scipy.signal

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.classes.*")

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# ======================================
# YAMNet HELPER FUNCTIONS
# ======================================
@st.cache_data(show_spinner=True)
def load_yamnet_model():
    """Carrega o modelo YAMNet do TensorFlow Hub."""
    model_handle = "https://tfhub.dev/google/yamnet/1"
    model = hub.load(model_handle)
    return model

@st.cache_data(show_spinner=True)
def load_class_map(yamnet_model):
    """Carrega o arquivo CSV de classes do YAMNet (class_map.csv)."""
    class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
    class_names = []
    import csv
    with tf.io.gfile.GFile(class_map_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row["display_name"])
    return class_names

def ensure_sample_rate(original_sr, waveform, desired_sr=16000):
    """Garante que o áudio esteja em desired_sr (16kHz). Caso contrário, resample."""
    if original_sr != desired_sr:
        desired_length = int(round(len(waveform) / original_sr * desired_sr))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sr, waveform

def normalize_waveform(waveform):
    """Normaliza o áudio para a faixa [-1.0, 1.0], se estiver em int16."""
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    elif waveform.dtype == np.float64:
        waveform = waveform.astype(np.float32)
    return waveform

def yamnet_inference(yamnet_model, waveform, class_names):
    """
    Faz a inferência com o YAMNet; retorna a classe inferida média,
    as pontuações por frame, embeddings e espectrograma log-mel.
    """
    scores, embeddings, spectrogram = yamnet_model(waveform)
    scores_np = scores.numpy()         # shape: (n_frames, 521)
    embeddings_np = embeddings.numpy() # shape: (n_frames, 1024)
    spectrogram_np = spectrogram.numpy()
    infered_class_idx = scores_np.mean(axis=0).argmax()
    infered_class = class_names[infered_class_idx]
    return infered_class, scores_np, embeddings_np, spectrogram_np

# ======================================
# CARREGAMENTO DOS ARQUIVOS DE ÁUDIO
# ======================================
def load_audio_files_from_zip(zip_file, desired_sr=16000):
    """
    Lê todos os arquivos de áudio de dentro de um zip,
    retorna uma lista de (waveform, label) + lista com nomes de classes.
    Cada subpasta do ZIP é interpretada como uma classe (similar a ImageFolder).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        data = []
        class_names = []
        for root, dirs, files in os.walk(tmp_dir):
            # Se len(dirs)=0, estamos na subpasta final => nome da classe
            if len(dirs) == 0:
                label = os.path.basename(root)
                class_names.append(label)
                for fname in files:
                    if fname.lower().endswith((".wav", ".mp3", ".ogg", ".flac", ".m4a")):
                        audio_path = os.path.join(root, fname)
                        try:
                            wav_data, sr = librosa.load(audio_path, sr=None, mono=True)
                            sr, wav_data = ensure_sample_rate(sr, wav_data, desired_sr)
                            wav_data = normalize_waveform(wav_data)
                            data.append((wav_data, label))
                        except Exception as e:
                            logging.warning(f"Falha ao carregar {audio_path}: {e}")
        unique_labels = sorted(list(set([x[1] for x in data])))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        final_data = []
        for wav_data, label in data:
            final_data.append((wav_data, label_to_idx[label]))
        return final_data, unique_labels

# ======================================
# TREINAMENTO / AVALIAÇÃO
# ======================================
def train_audio_classification(yamnet_model, class_map, data, unique_labels,
                               train_split, valid_split, epochs, learning_rate,
                               batch_size, use_weighted_loss, l2_lambda, patience, run_id):
    """
    Pipeline para treinar um classificador baseado nos embeddings do YAMNet.
    """
    random.shuffle(data)
    N = len(data)
    train_end = int(train_split * N)
    valid_end = int((train_split + valid_split) * N)

    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]

    st.write(f"Divisão: Treino={len(train_data)}, Validação={len(valid_data)}, Teste={len(test_data)}")

    # Extrair embeddings
    def extract_embeddings(dataset):
        X_list, y_list = [], []
        for (wav_data, label_idx) in dataset:
            waveform_tf = tf.constant(wav_data, tf.float32)
            _, scores_np, embeddings_np, spec_np = yamnet_inference(
                yamnet_model, waveform_tf, class_map
            )
            # Vamos usar a média ao longo dos frames
            emb_mean = embeddings_np.mean(axis=0)
            X_list.append(emb_mean)
            y_list.append(label_idx)
        X = np.vstack(X_list)
        y = np.array(y_list)
        return X, y

    st.write("Extraindo embeddings - Treino...")
    X_train, y_train = extract_embeddings(train_data)
    st.write("Extraindo embeddings - Validação...")
    X_val, y_val = extract_embeddings(valid_data)
    st.write("Extraindo embeddings - Teste...")
    X_test, y_test = extract_embeddings(test_data)

    # PCA de X_train
    visualize_embeddings(X_train, y_train, unique_labels, run_id)

    # Montar MLP
    from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
    num_classes = len(unique_labels)
    model_ff = models.Sequential()
    model_ff.add(layers.Input(shape=(1024,)))
    model_ff.add(layers.Dense(256, activation='relu',
                              kernel_regularizer=regularizers.l2(l2_lambda)))
    model_ff.add(layers.Dropout(0.5))
    model_ff.add(layers.Dense(num_classes, activation='softmax'))

    # Weighted?
    if use_weighted_loss:
        from collections import Counter
        ccount = Counter(y_train)
        class_weights = {c: 1.0/(ccount[c] if ccount[c]>0 else 1e-6) for c in range(num_classes)}
    else:
        class_weights = None

    model_ff.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    early_stop = callbacks.EarlyStopping(monitor='val_loss',
                                         patience=patience,
                                         restore_best_weights=True)
    history = model_ff.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    plot_loss_accuracy(history, run_id)

    # Avaliar no teste
    evaluate_on_test(model_ff, X_test, y_test, unique_labels, run_id)

    return model_ff

def visualize_embeddings(X, y, unique_labels, run_id):
    # PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=y, palette='Set2', legend='full')
    plt.title(f"Embeddings (PCA) - Run {run_id}")
    fname = f"embeddings_run{run_id}.png"
    plt.savefig(fname)
    st.image(fname, caption="PCA Embeddings", use_container_width=True)
    with open(fname, "rb") as f:
        st.download_button("Download Plot PCA", data=f, file_name=fname, mime="image/png")
    plt.close()

def plot_loss_accuracy(history, run_id):
    hist = history.history
    epochs_range = range(1, len(hist['loss'])+1)
    fig, ax = plt.subplots(1,2, figsize=(12,4))

    ax[0].plot(epochs_range, hist['loss'], label='Train')
    ax[0].plot(epochs_range, hist['val_loss'], label='Val')
    ax[0].set_title('Loss')
    ax[0].legend()

    ax[1].plot(epochs_range, hist['accuracy'], label='Train')
    ax[1].plot(epochs_range, hist['val_accuracy'], label='Val')
    ax[1].set_title('Accuracy')
    ax[1].legend()

    fname = f"training_curves_run{run_id}.png"
    plt.savefig(fname)
    st.image(fname, caption="Curvas de Treinamento", use_container_width=True)
    with open(fname, "rb") as f:
        st.download_button("Download Curvas", data=f, file_name=fname, mime="image/png")
    plt.close()

def evaluate_on_test(model, X_test, y_test, unique_labels, run_id):
    preds = model.predict(X_test)
    y_pred = np.argmax(preds, axis=1)

    # Matriz confusão
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels, fmt='.2f')
    ax.set_title("Matriz de Confusão Normalizada")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    fname = f"cm_run{run_id}.png"
    plt.savefig(fname)
    st.image(fname, caption="Matriz de Confusão (Normalizada)", use_container_width=True)
    with open(fname, "rb") as f:
        st.download_button("Download CM", data=f, file_name=fname, mime="image/png")
    plt.close()

    # Relatório
    rep = classification_report(y_test, y_pred, target_names=unique_labels, output_dict=True)
    df_rep = pd.DataFrame(rep).transpose()
    st.write("**Relatório de Classificação**")
    st.dataframe(df_rep)
    rep_filename = f"report_run{run_id}.csv"
    df_rep.to_csv(rep_filename)
    with open(rep_filename, "rb") as f:
        st.download_button("Download Report", data=f, file_name=rep_filename, mime="text/csv")

    # Se for binário
    if len(unique_labels)==2:
        probs = preds[:,1]
        aucv = roc_auc_score(y_test, probs)
        fpr, tpr, _ = roc_curve(y_test, probs)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={aucv:.3f}")
        ax.plot([0,1],[0,1],'k--')
        ax.set_title("ROC")
        ax.legend()
        fname = f"roc_run{run_id}.png"
        plt.savefig(fname)
        st.image(fname, caption="Curva ROC", use_container_width=True)
        with open(fname, "rb") as f:
            st.download_button("Download ROC", data=f, file_name=fname, mime="image/png")
        plt.close()

# ======================================
# CLUSTERING OPCIONAL
# ======================================
def cluster_audio_embeddings(X, y, unique_labels, run_id):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    km = KMeans(n_clusters=len(unique_labels), random_state=42).fit(X)
    y_km = km.labels_

    agglo = AgglomerativeClustering(n_clusters=len(unique_labels)).fit(X)
    y_ag = agglo.labels_

    fig, ax = plt.subplots(1,2, figsize=(12,5))
    ax[0].scatter(X_2d[:,0], X_2d[:,1], c=y_km, cmap='viridis')
    ax[0].set_title("KMeans")
    ax[1].scatter(X_2d[:,0], X_2d[:,1], c=y_ag, cmap='viridis')
    ax[1].set_title("Agglomerative")
    fname = f"clustering_run{run_id}.png"
    plt.savefig(fname)
    st.image(fname, caption="Clusterização", use_container_width=True)
    with open(fname, "rb") as f:
        st.download_button("Download Clusters", data=f, file_name=fname, mime="image/png")
    plt.close()

    ari_km = adjusted_rand_score(y, y_km)
    nmi_km = normalized_mutual_info_score(y, y_km)
    ari_ag = adjusted_rand_score(y, y_ag)
    nmi_ag = normalized_mutual_info_score(y, y_ag)
    st.write(f"KMeans: ARI={ari_km:.3f}, NMI={nmi_km:.3f}")
    st.write(f"Agglo: ARI={ari_ag:.3f}, NMI={nmi_ag:.3f}")

# ======================================
# STREAMLIT APP
# ======================================
def main():
    st.set_page_config(page_title="Classificação de Áudio - YAMNet", layout="wide")
    st.title("Classificação de Áudio (Qualidade da Água) usando YAMNet")

    st.sidebar.title("Configurações")
    train_split = st.sidebar.slider("Train Split", 0.5, 0.9, 0.7, 0.05)
    valid_split = st.sidebar.slider("Val Split", 0.05, 0.4, 0.15, 0.05)
    epochs = st.sidebar.slider("Epochs", 1, 100, 10)
    lr = st.sidebar.select_slider("Learning Rate", [0.1, 0.01, 0.001, 0.0001], 0.001)
    batch_size = st.sidebar.selectbox("Batch Size", [4,8,16,32,64], index=2)
    use_weighted = st.sidebar.checkbox("Weighted Loss?", value=False)
    l2_reg = st.sidebar.number_input("L2 Regularization", 0.0, 0.1, 0.01, 0.01)
    patience_val = st.sidebar.number_input("EarlyStop Patience", 1, 10, 3)

    zip_file = st.file_uploader("Envie o ZIP contendo subpastas=classes de áudio", type=["zip"])
    if zip_file is not None:
        # Carrega YAMNet
        st.write("Carregando modelo YAMNet...")
        yam_model = load_yamnet_model()
        class_map = load_class_map(yam_model)
        st.write(f"YAMNet OK. Tamanho do class_map={len(class_map)} (AudioSet)")

        data, labels = load_audio_files_from_zip(zip_file)
        st.write(f"Carregados {len(data)} exemplos de áudio.")
        st.write(f"Classes encontradas: {labels}")

        if len(data) == 0:
            st.warning("Nenhum arquivo de áudio foi lido do ZIP.")
            return

        run_id = str(uuid.uuid4())[:8]
        st.write(f"Treinando com ID={run_id}")

        # Treino
        model_ff = train_audio_classification(
            yamnet_model=yam_model,
            class_map=class_map,
            data=data,
            unique_labels=labels,
            train_split=train_split,
            valid_split=valid_split,
            epochs=epochs,
            learning_rate=lr,
            batch_size=batch_size,
            use_weighted_loss=use_weighted,
            l2_lambda=l2_reg,
            patience=patience_val,
            run_id=run_id
        )

        # Cluster?
        do_cluster = st.checkbox("Fazer clusterização das embeddings de TODO o dataset?", value=False)
        if do_cluster:
            from tqdm import tqdm
            X_all, y_all = [], []
            for (wav_data, label_idx) in tqdm(data):
                tfwav = tf.constant(wav_data, tf.float32)
                _, sc_np, emb_np, sp_np = yamnet_inference(yam_model, tfwav, class_map)
                X_all.append(emb_np.mean(axis=0))
                y_all.append(label_idx)
            X_all = np.vstack(X_all)
            y_all = np.array(y_all)
            cluster_audio_embeddings(X_all, y_all, labels, run_id)

if __name__ == "__main__":
    main()

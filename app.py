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
    Faz a inferência com o YAMNet; retorna as pontuações por frame,
    o embedding e o espectrograma log-mel.
    """
    scores, embeddings, spectrogram = yamnet_model(waveform)
    scores_np = scores.numpy()         # shape: (n_frames, 521)
    embeddings_np = embeddings.numpy() # shape: (n_frames, 1024)
    spectrogram_np = spectrogram.numpy()
    infered_class_idx = scores_np.mean(axis=0).argmax()
    infered_class = class_names[infered_class_idx]
    return infered_class, scores_np, embeddings_np, spectrogram_np

# ======================================
# DATASET / DATA LOADING
# ======================================
def load_audio_files_from_zip(zip_file, desired_sr=16000):
    """
    Lê todos os arquivos de áudio de dentro de um zip,
    retorna uma lista de (waveform, label) + labels.
    Supõe-se que cada subpasta do ZIP seja uma classe (similar a ImageFolder).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        # Percorre subpastas (cada subpasta = classe)
        data = []
        class_names = []
        for root, dirs, files in os.walk(tmp_dir):
            if len(dirs) == 0:  # significa que root é nível "subpasta"
                label = os.path.basename(root)
                class_names.append(label)
                for fname in files:
                    if fname.lower().endswith((".wav", ".mp3", ".ogg", ".flac", ".m4a")):
                        audio_path = os.path.join(root, fname)
                        try:
                            sr, wav_data = None, None
                            # Carrega via librosa
                            wav_data, sr = librosa.load(audio_path, sr=None, mono=True)
                            sr, wav_data = ensure_sample_rate(sr, wav_data, desired_sr)
                            wav_data = normalize_waveform(wav_data)
                            data.append((wav_data, label))
                        except Exception as e:
                            logging.warning(f"Falha ao carregar {audio_path}: {e}")
        # Convert label -> int idx
        unique_labels = sorted(list(set([x[1] for x in data])))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        final_data = []
        for wav_data, label in data:
            final_data.append((wav_data, label_to_idx[label]))
        return final_data, unique_labels

# ======================================
# SPLIT DATA, TRAIN, EVALUATE
# ======================================
def train_audio_classification(yamnet_model, class_map, data, unique_labels,
                               train_split, valid_split, epochs, learning_rate,
                               batch_size, use_weighted_loss, l2_lambda, patience, run_id):
    """
    Pipeline principal para treinar um 'modelo' de áudio com
    comportamento análogo ao do exemplo de imagens (ResNet...), mas
    aqui utilizamos YAMNet como extrator de embeddings e depois
    treinamos um classificador feed-forward simples (MLP ou similar).

    - `data`: lista de (waveform, label_idx).
    - `unique_labels`: lista de strings (nomes das classes).
    """
    # Emulando data augmentation (ex.: pitch shift) - não implementado aqui
    # Em seu lugar, poderíamos usar audiomentations ou algo similar.

    # Embaralhar e dividir
    random.shuffle(data)
    N = len(data)
    train_end = int(train_split * N)
    valid_end = int((train_split + valid_split) * N)

    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]

    st.write(f"Tamanho: Treino={len(train_data)}, Validação={len(valid_data)}, Teste={len(test_data)}.")

    # Extração de embeddings
    def extract_embeddings(dataset):
        """
        Extrai embeddings e retorna X, y.
        X: np.array de shape (num_samples, 1024) se YAMNet embedding
        y: rótulos
        """
        X_list, y_list = [], []
        for (wav_data, label_idx) in dataset:
            # Passa o wav_data no modelo
            # Formato esperado: tf.float32 tensor [n_samples], normalizado
            waveform_tf = tf.constant(wav_data, tf.float32)
            infered_class, scores_np, embeddings_np, spec_np = yamnet_inference(
                yamnet_model, waveform_tf, class_map
            )
            # Embedding médio ao longo dos frames
            embedding_mean = embeddings_np.mean(axis=0)
            X_list.append(embedding_mean)
            y_list.append(label_idx)
        X = np.vstack(X_list)
        y = np.array(y_list)
        return X, y

    st.write("Extraindo embeddings do conjunto de Treino...")
    X_train, y_train = extract_embeddings(train_data)
    st.write("Extraindo embeddings do conjunto de Validação...")
    X_valid, y_valid = extract_embeddings(valid_data)
    st.write("Extraindo embeddings do conjunto de Teste...")
    X_test, y_test = extract_embeddings(test_data)

    # Visualizar embeddings em PCA
    visualize_embeddings(X_train, y_train, unique_labels, run_id)

    # Montar classificador MLP em TensorFlow/Keras ou sklearn
    from tensorflow.keras import layers, models, optimizers, callbacks

    num_classes = len(unique_labels)
    model_ff = models.Sequential()
    model_ff.add(layers.Input(shape=(1024,)))
    model_ff.add(layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)))
    model_ff.add(layers.Dropout(0.5))
    model_ff.add(layers.Dense(num_classes, activation='softmax'))

    # Weighted loss
    if use_weighted_loss:
        from collections import Counter
        counts = Counter(y_train)
        class_weights = {}
        for c in range(num_classes):
            class_weights[c] = 1.0 / (counts[c] if c in counts else 1e-6)
    else:
        class_weights = None

    model_ff.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    history = model_ff.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=1
    )

    # Plot das curvas
    plot_loss_accuracy(history, run_id)

    # Avaliar no teste
    evaluate_on_test(model_ff, X_test, y_test, unique_labels, run_id)

    # Retorna o classificador final
    return model_ff

def visualize_embeddings(X, y, unique_labels, run_id):
    """PCA 2D e plot."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='Set2', legend='full')
    plt.title(f"Embeddings PCA - Run {run_id}")
    embeddings_pca_filename = f"embeddings_pca_run{run_id}.png"
    plt.savefig(embeddings_pca_filename)
    st.image(embeddings_pca_filename, caption='Embeddings PCA', use_container_width=True)
    with open(embeddings_pca_filename, "rb") as file:
        st.download_button(
            "Download PCA Embeddings Plot",
            data=file,
            file_name=embeddings_pca_filename,
            mime="image/png"
        )
    plt.close()

def plot_loss_accuracy(history, run_id):
    """Plota as curvas de loss e accuracy."""
    hist = history.history
    epochs_range = range(1, len(hist['loss'])+1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs_range, hist['loss'], label='Train Loss')
    ax[0].plot(epochs_range, hist['val_loss'], label='Valid Loss')
    ax[0].legend()
    ax[0].set_title("Loss x Épocas")

    ax[1].plot(epochs_range, hist['accuracy'], label='Train Acc')
    ax[1].plot(epochs_range, hist['val_accuracy'], label='Valid Acc')
    ax[1].legend()
    ax[1].set_title("Accuracy x Épocas")

    plt.tight_layout()
    plot_filename = f"loss_accuracy_run{run_id}.png"
    plt.savefig(plot_filename)
    st.image(plot_filename, caption='Curvas de Treinamento', use_container_width=True)

    with open(plot_filename, "rb") as file:
        st.download_button(
            "Download Curvas de Treinamento",
            data=file,
            file_name=plot_filename,
            mime="image/png"
        )
    plt.close()

def evaluate_on_test(model_ff, X_test, y_test, unique_labels, run_id):
    """Avalia no teste e plota matriz de confusão, etc."""
    preds = model_ff.predict(X_test)
    pred_classes = np.argmax(preds, axis=1)

    # Matriz de Confusão
    cm = confusion_matrix(y_test, pred_classes, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    ax.set_title('Matriz de Confusão Normalizada')
    plt.tight_layout()
    cm_filename = f"cm_run{run_id}.png"
    plt.savefig(cm_filename)
    st.image(cm_filename, caption='Matriz de Confusão', use_container_width=True)
    with open(cm_filename, "rb") as file:
        st.download_button(
            "Download Matriz de Confusão",
            data=file,
            file_name=cm_filename,
            mime="image/png"
        )
    plt.close()

    # Relatório de Classificação
    report = classification_report(y_test, pred_classes, target_names=unique_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write("**Relatório de Classificação**:")
    st.dataframe(report_df)
    cr_filename = f"classification_report_run{run_id}.csv"
    report_df.to_csv(cr_filename, index=True)
    with open(cr_filename, "rb") as file:
        st.download_button(
            "Download Relatório de Classificação",
            data=file,
            file_name=cr_filename,
            mime="text/csv"
        )

    # AUC-ROC binária ou multiclasse
    # ...
    # Exemplo simplificado (somente se 2 classes)
    if len(unique_labels) == 2:
        y_probs = preds[:,1]
        from sklearn.metrics import roc_auc_score, roc_curve
        auc = roc_auc_score(y_test, y_probs)
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        ax.plot([0,1],[0,1], 'k--')
        ax.set_title("ROC Curve")
        ax.legend()
        roc_filename = f"roc_run{run_id}.png"
        plt.savefig(roc_filename)
        st.image(roc_filename, caption='Curva ROC', use_container_width=True)
        with open(roc_filename, "rb") as file:
            st.download_button(
                "Download Curva ROC",
                data=file,
                file_name=roc_filename,
                mime="image/png"
            )
        plt.close()

# ======================================
# CLUSTERING
# ======================================
def cluster_audio_embeddings(X, y, unique_labels, run_id):
    """Faz clustering (KMeans, Agglo) e plota PCA 2D."""
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    km = KMeans(n_clusters=len(unique_labels), random_state=42).fit(X)
    clusters_km = km.labels_

    agglo = AgglomerativeClustering(n_clusters=len(unique_labels)).fit(X)
    clusters_agglo = agglo.labels_

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sc1 = ax[0].scatter(X_2d[:,0], X_2d[:,1], c=clusters_km, cmap='viridis')
    ax[0].set_title("KMeans Clustering")

    sc2 = ax[1].scatter(X_2d[:,0], X_2d[:,1], c=clusters_agglo, cmap='viridis')
    ax[1].set_title("Agglomerative Clustering")

    fig.suptitle(f"Clustering - Run {run_id}")
    plt.tight_layout()
    cluster_filename = f"clustering_run{run_id}.png"
    plt.savefig(cluster_filename)
    st.image(cluster_filename, caption='Clusterização', use_container_width=True)
    with open(cluster_filename, "rb") as file:
        st.download_button(
            "Download Clustering Plot",
            data=file,
            file_name=cluster_filename,
            mime="image/png"
        )
    plt.close()

    # ARI, NMI
    ari_km = adjusted_rand_score(y, clusters_km)
    nmi_km = normalized_mutual_info_score(y, clusters_km)
    ari_ag = adjusted_rand_score(y, clusters_agglo)
    nmi_ag = normalized_mutual_info_score(y, clusters_agglo)

    st.write(f"**KMeans**: ARI={ari_km:.3f}, NMI={nmi_km:.3f}")
    st.write(f"**Agglomerative**: ARI={ari_ag:.3f}, NMI={nmi_ag:.3f}")

# ======================================
# MAIN APP
# ======================================
def main():
    st.set_page_config(page_title="Classificação de Áudio com YAMNet", layout="wide")
    st.title("Classificação de Áudio com YAMNet - Exemplo Estilo ResNet")

    st.sidebar.header("Parâmetros de Treinamento")
    train_split = st.sidebar.slider("Train Split", 0.5, 0.9, 0.7, 0.05)
    valid_split = st.sidebar.slider("Validation Split", 0.05, 0.4, 0.15, 0.05)
    epochs = st.sidebar.slider("Epochs", 1, 100, 10)
    learning_rate = st.sidebar.select_slider("Learning Rate", [0.1, 0.01, 0.001, 0.0001], value=0.001)
    batch_size = st.sidebar.selectbox("Batch Size", [4,8,16,32,64], index=2)
    use_weighted_loss = st.sidebar.checkbox("Weighted Loss?", value=False)
    l2_lambda = st.sidebar.number_input("L2 Regularization", 0.0, 0.1, 0.01, 0.01)
    patience = st.sidebar.number_input("Early Stopping Patience", 1, 10, 3)

    # Uploader
    zip_file = st.file_uploader("Upload do ZIP contendo subpastas = classes de áudio", type=["zip"])
    if zip_file:
        # Carregar modelo YAMNet + Classes
        st.write("Carregando YAMNet...")
        yamnet_model = load_yamnet_model()
        class_map = load_class_map(yamnet_model)
        st.write(f"YAMNet carregado! Classes={len(class_map)} do AudioSet.")

        data, unique_labels = load_audio_files_from_zip(zip_file)
        st.write(f"Número total de amostras carregadas: {len(data)}")
        st.write(f"Classes encontradas: {unique_labels} (Total={len(unique_labels)})")

        if len(data) > 0:
            run_id = str(uuid.uuid4())[:8]
            st.write(f"Iniciando Treino - Run={run_id}")
            model_ff = train_audio_classification(
                yamnet_model, class_map, data, unique_labels,
                train_split, valid_split,
                epochs, learning_rate, batch_size,
                use_weighted_loss, l2_lambda, patience,
                run_id
            )

            # Cluster embeddings do dataset inteiro?
            st.write("**Clusterização de todo o dataset** (opcional).")
            do_cluster = st.checkbox("Fazer clusterização das embeddings?", value=False)
            if do_cluster:
                from tqdm import tqdm
                # Extrair embeddings para o dataset completo
                X_all, y_all = [], []
                for (wav_data, label_idx) in tqdm(data):
                    waveform_tf = tf.constant(wav_data, tf.float32)
                    _, scores_np, embeddings_np, spec_np = yamnet_inference(yamnet_model, waveform_tf, class_map)
                    emb_mean = embeddings_np.mean(axis=0)
                    X_all.append(emb_mean)
                    y_all.append(label_idx)
                X_all = np.vstack(X_all)
                y_all = np.array(y_all)
                cluster_audio_embeddings(X_all, y_all, unique_labels, run_id)
        else:
            st.warning("Nenhum áudio pôde ser carregado do ZIP.")

if __name__ == "__main__":
    main()

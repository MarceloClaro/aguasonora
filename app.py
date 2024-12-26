import os
import zipfile
import shutil
import tempfile
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import streamlit as st
import gc
import logging
import io
import tensorflow as tf
import tensorflow_hub as hub
from scipy.io import wavfile
import scipy.signal
from datetime import datetime
import librosa
import librosa.display
import requests  # Para download de arquivos de áudio

# Instalar dependências necessárias
# Certifique-se de que estas linhas estão no início do seu script ou na configuração do ambiente
# pip install pydub numba==0.48 librosa music21 scikit-learn

# Suprimir avisos relacionados ao torch.classes
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.classes.*")

# Definir o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurações para gráficos mais bonitos
sns.set_style('whitegrid')

# Definir seed para reprodutibilidade
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Garantir reprodutibilidade
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Definir a seed

@st.cache_resource
def load_yamnet_model():
    """
    Carrega o modelo YAMNet do TF Hub.
    """
    yam_model = hub.load('https://tfhub.dev/google/yamnet/1')
    return yam_model

def ensure_sample_rate(original_sr, waveform, desired_sr=16000):
    """
    Resample se não estiver em 16 kHz.
    """
    if original_sr != desired_sr:
        desired_length = int(round(float(len(waveform)) / original_sr * desired_sr))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sr, waveform

def add_noise(waveform, noise_factor=0.005):
    """
    Adiciona ruído branco ao áudio.
    """
    noise = np.random.randn(len(waveform))
    augmented_waveform = waveform + noise_factor * noise
    augmented_waveform = augmented_waveform.astype(np.float32)
    return augmented_waveform

def time_stretch(waveform, rate=1.1):
    """
    Estica o tempo do áudio.
    """
    return librosa.effects.time_stretch(waveform, rate)

def pitch_shift(waveform, sr, n_steps=2):
    """
    Muda a altura do áudio.
    """
    return librosa.effects.pitch_shift(waveform, sr, n_steps)

def perform_data_augmentation(waveform, sr, augmentation_methods, rate=1.1, n_steps=2):
    """
    Aplica data augmentation no áudio.
    """
    augmented_waveforms = [waveform]
    for method in augmentation_methods:
        if method == 'Add Noise':
            augmented_waveforms.append(add_noise(waveform))
        elif method == 'Time Stretch':
            try:
                stretched = time_stretch(waveform, rate=rate)
                augmented_waveforms.append(stretched)
            except Exception as e:
                st.warning(f"Erro ao aplicar Time Stretch: {e}")
        elif method == 'Pitch Shift':
            try:
                shifted = pitch_shift(waveform, sr, n_steps=n_steps)
                augmented_waveforms.append(shifted)
            except Exception as e:
                st.warning(f"Erro ao aplicar Pitch Shift: {e}")
    return augmented_waveforms

def extract_yamnet_embeddings(yamnet_model, audio_path):
    """
    Extrai embeddings usando o modelo YAMNet para um arquivo de áudio.
    Retorna a classe predita e a média dos embeddings das frames.
    """
    basename_audio = os.path.basename(audio_path)
    try:
        sr_orig, wav_data = wavfile.read(audio_path)
        st.write(f"Processando {basename_audio}: Sample Rate = {sr_orig}, Shape = {wav_data.shape}, Dtype = {wav_data.dtype}")
        
        # Verificar se está estéreo
        if wav_data.ndim > 1:
            # Converter para mono
            wav_data = wav_data.mean(axis=1)
            st.write(f"Convertido para mono: Shape = {wav_data.shape}")
        
        # Normalizar para [-1, 1] ou verificar se já está normalizado
        if wav_data.dtype.kind == 'i':
            # Dados inteiros
            max_val = np.iinfo(wav_data.dtype).max
            waveform = wav_data / max_val
            st.write(f"Normalizado de inteiros para float: max_val = {max_val}")
        elif wav_data.dtype.kind == 'f':
            # Dados float
            waveform = wav_data
            # Verificar se os dados estão fora do intervalo [-1.0, 1.0]
            if np.max(waveform) > 1.0 or np.min(waveform) < -1.0:
                waveform = waveform / np.max(np.abs(waveform))
                st.write("Normalizado para o intervalo [-1.0, 1.0]")
        else:
            raise ValueError(f"Tipo de dado do áudio não suportado: {wav_data.dtype}")
    
        # Garantir que é float32
        waveform = waveform.astype(np.float32)
    
        # Ajustar sample rate
        sr, waveform = ensure_sample_rate(sr_orig, waveform)
        st.write(f"Sample Rate ajustado: {sr}")
    
        # Executar o modelo YAMNet
        # yamnet_model retorna: scores, embeddings, spectrogram
        scores, embeddings, spectrogram = yamnet_model(waveform)
        st.write(f"Embeddings extraídos: Shape = {embeddings.shape}")
    
        # scores.shape = [frames, 521]
        scores_np = scores.numpy()
        mean_scores = scores_np.mean(axis=0)  # média por frame
        pred_class = mean_scores.argmax()
        st.write(f"Classe predita pelo YAMNet: {pred_class}")
    
        # Calcular a média dos embeddings das frames para obter um embedding fixo
        mean_embedding = embeddings.numpy().mean(axis=0)  # Shape: (1024,)
        st.write(f"Média dos embeddings das frames: Shape = {mean_embedding.shape}")
    
        return pred_class, mean_embedding
    except Exception as e:
        st.error(f"Erro ao processar {basename_audio}: {e}")
        return -1, None

def balance_classes(X, y, method):
    """
    Balanceia as classes usando oversampling ou undersampling.
    """
    if method == 'Oversample':
        smote = SMOTE(random_state=42)
        X_bal, y_bal = smote.fit_resample(X, y)
    elif method == 'Undersample':
        rus = RandomUnderSampler(random_state=42)
        X_bal, y_bal = rus.fit_resample(X, y)
    else:
        X_bal, y_bal = X, y
    return X_bal, y_bal

def train_audio_classifier(X_train, y_train, X_val, y_val, input_dim, num_classes, epochs, learning_rate, batch_size, l2_lambda, patience):
    """
    Treina um classificador simples em PyTorch com os embeddings extraídos.
    """
    # Converter para tensores
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Criar DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Definir um classificador simples
    class AudioClassifier(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(AudioClassifier, self).__init__()
            self.fc1 = nn.Linear(input_dim, 256)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    classifier = AudioClassifier(input_dim, num_classes).to(device)

    # Definir a função de perda e otimizador com L2 regularization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    # Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None

    # Inicializar listas para métricas
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    progress_bar = st.progress(0)
    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        # Validação
        classifier.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = classifier(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc.item())

        # Atualizar a barra de progresso
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)

        # Exibir métricas do epoch
        st.write(f"### Época {epoch+1}/{epochs}")
        st.write(f"**Treino - Perda:** {epoch_loss:.4f}, **Acurácia:** {epoch_acc:.4f}")
        st.write(f"**Validação - Perda:** {val_epoch_loss:.4f}, **Acurácia:** {val_epoch_acc:.4f}")

        # Early Stopping
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
            best_model_wts = classifier.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                st.write("Early stopping!")
                break

    # Carregar os melhores pesos do modelo se houver
    if best_model_wts is not None:
        classifier.load_state_dict(best_model_wts)

    # Plotar métricas
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Perda
    ax[0].plot(range(1, len(train_losses)+1), train_losses, label='Treino')
    ax[0].plot(range(1, len(val_losses)+1), val_losses, label='Validação')
    ax[0].set_title('Perda por Época')
    ax[0].set_xlabel('Épocas')
    ax[0].set_ylabel('Perda')
    ax[0].legend()

    # Acurácia
    ax[1].plot(range(1, len(train_accuracies)+1), train_accuracies, label='Treino')
    ax[1].plot(range(1, len(val_accuracies)+1), val_accuracies, label='Validação')
    ax[1].set_title('Acurácia por Época')
    ax[1].set_xlabel('Épocas')
    ax[1].set_ylabel('Acurácia')
    ax[1].legend()

    st.pyplot(fig)
    plt.close(fig)

    # Avaliação no conjunto de validação
    classifier.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = classifier(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Relatório de Classificação
    st.write("### Relatório de Classificação")
    target_names = [f"Classe {cls}" for cls in set(y_train)]
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    # Matriz de Confusão
    st.write("### Matriz de Confusão")
    cm = confusion_matrix(all_labels, all_preds)
    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # Curva ROC e AUC
    st.write("### Curva ROC")
    if num_classes > 1:
        # Para múltiplas classes, utilizamos One-vs-Rest
        from sklearn.preprocessing import label_binarize

        y_test_binarized = label_binarize(all_labels, classes=range(num_classes))
        y_pred_binarized = label_binarize(all_preds, classes=range(num_classes))

        if y_test_binarized.shape[1] > 1:
            fpr = dict()
            tpr = dict()
            roc_auc_dict = dict()
            for i in range(num_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
                roc_auc_dict[i] = auc(fpr[i], tpr[i])

            # Plotar Curva ROC para cada classe
            fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
            colors = sns.color_palette("hsv", num_classes)
            for i, color in zip(range(num_classes), colors):
                ax_roc.plot(fpr[i], tpr[i], color=color, lw=2,
                           label=f'Classe {i} (AUC = {roc_auc_dict[i]:0.2f})')

            ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
            plt.close(fig_roc)
        else:
            st.warning("Curva ROC não disponível para uma única classe após binarização.")
    else:
        st.warning("Curva ROC não disponível para uma única classe.")

    # F1-Score
    st.write("### F1-Score por Classe")
    f1_scores = f1_score(all_labels, all_preds, average=None)
    f1_df = pd.DataFrame({'Classe': target_names, 'F1-Score': f1_scores})
    st.write(f1_df)

    return classifier

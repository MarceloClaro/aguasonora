import os
import zipfile
import tempfile
import shutil
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
import streamlit as st
import io

# Definir o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    """Define a seed para garantir a reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Função para converter áudio em imagens (espectrogramas)
def audio_to_spectrogram(audio_path, output_dir, sr=22050, img_size=(224, 224)):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        plt.figure(figsize=(2.24, 2.24), dpi=100)
        librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
        plt.axis('off')

        temp_image = os.path.join(output_dir, os.path.basename(audio_path).replace('.wav', '.png'))
        plt.savefig(temp_image, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Redimensionar a imagem
        image = Image.open(temp_image).resize(img_size)
        image.save(temp_image)
    except Exception as e:
        print(f"Erro ao processar {audio_path}: {e}")

# Dataset personalizado para espectrogramas
class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformações para treino e teste
train_transforms = transforms.Compose([
    transforms.RandomApply([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
    ], p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Função para criar e treinar o modelo
def train_model(image_dir, num_classes, model_name, epochs, learning_rate, batch_size, train_split, valid_split):
    set_seed(42)

    # Listar imagens e classes
    classes = sorted(os.listdir(image_dir))
    image_paths = []
    labels = []
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(image_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, image_name))
            labels.append(label_idx)

    # Dividir os dados em treino, validação e teste
    indices = list(range(len(image_paths)))
    random.shuffle(indices)
    train_end = int(train_split * len(indices))
    valid_end = int((train_split + valid_split) * len(indices))

    train_indices = indices[:train_end]
    valid_indices = indices[train_end:valid_end]
    test_indices = indices[valid_end:]

    train_dataset = SpectrogramDataset(
        [image_paths[i] for i in train_indices], [labels[i] for i in train_indices], transform=train_transforms
    )
    valid_dataset = SpectrogramDataset(
        [image_paths[i] for i in valid_indices], [labels[i] for i in valid_indices], transform=test_transforms
    )
    test_dataset = SpectrogramDataset(
        [image_paths[i] for i in test_indices], [labels[i] for i in test_indices], transform=test_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Carregar o modelo pré-treinado
    if model_name == 'ResNet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'ResNet50':
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError("Modelo não suportado.")

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Treinamento
    best_model_wts = None
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f"Epoch {epoch+1}/{epochs}: Loss {epoch_loss:.4f} Acc {epoch_acc:.4f}")

        # Validação
        model.eval()
        val_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)

        val_acc = val_running_corrects.double() / len(valid_dataset)
        print(f"Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

    # Avaliação no conjunto de teste
    model.load_state_dict(best_model_wts)
    model.eval()
    test_corrects = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = test_corrects.double() / len(test_dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Matriz de Confusão e Relatório de Classificação
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(all_labels, all_preds))

    return model, classes

def main():
    st.title("Classificação de Áudio usando Espectrogramas")
    st.sidebar.header("Configurações de Treinamento")

    # Configurações
    model_name = st.sidebar.selectbox("Modelo Pré-treinado", ["ResNet18", "ResNet50"])
    num_classes = st.sidebar.number_input("Número de Classes", min_value=2, step=1, value=5)
    epochs = st.sidebar.slider("Épocas", min_value=1, max_value=50, value=10)
    learning_rate = st.sidebar.select_slider("Taxa de Aprendizado", options=[0.1, 0.01, 0.001, 0.0001], value=0.001)
    batch_size = st.sidebar.selectbox("Tamanho do Lote", options=[4, 8, 16, 32], index=1)
    train_split = st.sidebar.slider("Divisão de Treinamento", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
    valid_split = st.sidebar.slider("Divisão de Validação", min_value=0.05, max_value=0.4, value=0.15, step=0.05)

    # Upload do arquivo ZIP
    zip_file = st.file_uploader("Faça upload do arquivo ZIP com áudios", type=["zip"])

    if zip_file:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        audio_dir = os.path.join(temp_dir, "audios")
        spectrogram_dir = os.path.join(temp_dir, "spectrograms")
        os.makedirs(spectrogram_dir, exist_ok=True)

        # Converter áudios em espectrogramas
        for class_name in os.listdir(audio_dir):
            class_dir = os.path.join(audio_dir, class_name)
            output_class_dir = os.path.join(spectrogram_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)

            for audio_file in os.listdir(class_dir):
                audio_to_spectrogram(os.path.join(class_dir, audio_file), output_class_dir)

        # Treinar o modelo
        model, classes = train_model(spectrogram_dir, num_classes, model_name, epochs, learning_rate, batch_size, train_split, valid_split)

        # Opção para baixar o modelo treinado
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        st.download_button("Download do Modelo", data=buffer, file_name="modelo_treinado.pth", mime="application/octet-stream")

if __name__ == "__main__":
    main()

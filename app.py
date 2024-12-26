import os
import zipfile
import shutil
import tempfile
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
# Importar SOMENTE modelos que existem na versão atual
from torchvision.models import resnet18, resnet50, densenet121

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
import streamlit as st
import gc
import warnings
import io
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.classes.*")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sns.set_style("whitegrid")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    st.title("Classificação de Imagens (Sem fcn_resnet50)")
    st.write("Exemplo simplificado, usando somente ResNet e DenseNet para classificação.")

    # Sidebar
    num_classes = st.sidebar.number_input("Número de Classes", min_value=2, value=2, step=1)
    model_choice = st.sidebar.selectbox("Escolha o modelo", ["ResNet18","ResNet50","DenseNet121"])
    epochs = st.sidebar.slider("Épocas", 1, 20, 5)
    learning_rate = st.sidebar.select_slider("Learning Rate", options=[1e-2,1e-3,1e-4], value=1e-3)
    batch_size = st.sidebar.selectbox("Batch Size",[8,16,32], index=1)

    zip_file = st.file_uploader("Faça upload de um .zip com subpastas => classes (ex: agua gelada/, agua quente/)", type=["zip"])
    if not zip_file:
        st.info("Aguardando upload do .zip contendo as imagens")
        return

    # Extração do zip
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "data.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        data_dir = tmpdir
        st.write("Treinando modelo com dataset em:", data_dir)

        try:
            # Exemplo de uso do ImageFolder
            full_dataset = datasets.ImageFolder(root=data_dir, transform=transforms.ToTensor())
            if len(full_dataset.classes)<num_classes:
                st.error(f"Pastas encontradas = {full_dataset.classes} (total={len(full_dataset.classes)}), mas você pediu num_classes={num_classes}")
                return

            # Dividir em train/test
            size = len(full_dataset)
            indices = list(range(size))
            random.shuffle(indices)
            train_end = int(0.8*size)
            train_idx = indices[:train_end]
            test_idx  = indices[train_end:]

            train_ds = torch.utils.data.Subset(full_dataset, train_idx)
            test_ds  = torch.utils.data.Subset(full_dataset, test_idx)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            # Carrega o modelo
            if model_choice=="ResNet18":
                model = resnet18(weights="IMAGENET1K_V1")
                # Muda a última camada
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_classes)
            elif model_choice=="ResNet50":
                model = resnet50(weights="IMAGENET1K_V1")
                # Muda a última camada
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_classes)
            else:
                model = densenet121(weights="IMAGENET1K_V1")
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, num_classes)

            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            st.write("Classes detectadas:", full_dataset.classes)
            st.write("Treinando...")
            for ep in range(epochs):
                model.train()
                running_loss = 0.0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()*images.size(0)
                epoch_loss = running_loss/len(train_ds)
                st.write(f"Época {ep+1}/{epochs}, Loss={epoch_loss:.4f}")

            st.write("Avaliando no Teste...")
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs,1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            rep = classification_report(all_labels, all_preds, target_names=full_dataset.classes, output_dict=True)
            st.write(pd.DataFrame(rep).transpose())

        except FileNotFoundError as e:
            st.error(f"Erro: {str(e)}\nVerifique se há arquivos .jpg/.png dentro das pastas no .zip")
        except Exception as ex:
            st.error(f"Erro inesperado: {str(ex)}")

if __name__ == "__main__":
    set_seed(42)
    main()

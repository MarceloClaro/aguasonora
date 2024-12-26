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
from torchvision.models import resnet18, resnet50, densenet121
from torchvision.models import ResNet18_Weights, ResNet50_Weights, DenseNet121_Weights
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA

import streamlit as st
import gc
import logging
import base64

# ==== Bibliotecas para Grad-CAM com torchcam:
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask

# Transformações de imagem
from torchvision.transforms.functional import to_pil_image
import cv2
import io
import warnings
from datetime import datetime

# Supressão de avisos relacionados ao torch.classes
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.classes.*")

# Definir dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurações gerais de plots
sns.set_style('whitegrid')

def set_seed(seed):
    """Define uma seed para garantir reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Garantindo consistência
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Transforms para data augmentation e teste
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomApply([
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.RandomAffine(degrees=0, shear=10),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    ], p=0.5),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Dataset customizado
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Dataset customizado para Segmentação
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

def seed_worker(worker_id):
    """Garante reprodutibilidade nos DataLoaders."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def visualize_data(dataset, classes):
    """Mostra algumas imagens do dataset."""
    st.write("Visualização de algumas imagens do dataset:")
    fig, axes = plt.subplots(1, 10, figsize=(20, 4))
    for i in range(10):
        idx = np.random.randint(len(dataset))
        image, label = dataset[idx]
        image = np.array(image)
        axes[i].imshow(image)
        axes[i].set_title(classes[label])
        axes[i].axis('off')
    st.pyplot(fig)
    plt.close(fig)

def plot_class_distribution(dataset, classes):
    """Mostra contagem de classes no dataset."""
    labels = [label for _, label in dataset]
    df = pd.DataFrame({'Classe': labels})
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Classe', data=df, ax=ax, palette="Set2", hue='Classe', dodge=False)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.get_legend().remove()
    class_counts = df['Classe'].value_counts().sort_index()
    for i, count in enumerate(class_counts):
        ax.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    ax.set_title("Distribuição das Classes (Quantidade de Imagens)")
    ax.set_xlabel("Classes")
    ax.set_ylabel("Número de Imagens")
    st.pyplot(fig)
    plt.close(fig)

def get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False):
    """Retorna modelo pré-treinado para classificação."""
    if model_name == 'ResNet18':
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    elif model_name == 'ResNet50':
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    elif model_name == 'DenseNet121':
        weights = DenseNet121_Weights.DEFAULT
        model = densenet121(weights=weights)
    else:
        st.error("Modelo não suportado.")
        return None

    # Congelar ou não
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    # Ajustar camadas de saída
    if model_name.startswith('ResNet'):
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_name.startswith('DenseNet'):
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        st.error("Modelo não suportado (camada final).")
        return None

    model = model.to(device)
    return model

def get_segmentation_model(num_classes, fine_tune=False):
    """Retorna modelo para segmentação (FCN-ResNet50)."""
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights)
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model = model.to(device)
    return model

def apply_transforms_and_get_embeddings(dataset, model, transform, batch_size=16):
    """Extrai embeddings das imagens + Data Augmentation."""
    def pil_collate_fn(batch):
        images, labels = zip(*batch)
        return list(images), torch.tensor(labels)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pil_collate_fn)
    embeddings_list = []
    labels_list = []
    file_paths_list = []
    augmented_images_list = []

    # Remove a última camada
    model_embedding = nn.Sequential(*list(model.children())[:-1])
    model_embedding.eval()
    model_embedding.to(device)

    indices = dataset.indices if hasattr(dataset, 'indices') else list(range(len(dataset)))
    index_pointer = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images_augmented = [transform(img) for img in images]
            images_augmented = torch.stack(images_augmented).to(device)
            embeddings = model_embedding(images_augmented)
            embeddings = embeddings.view(embeddings.size(0), -1).cpu().numpy()
            embeddings_list.extend(embeddings)
            labels_list.extend(labels.numpy())
            augmented_images_list.extend([img.permute(1, 2, 0).numpy() for img in images_augmented.cpu()])
            if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'samples'):
                batch_indices = indices[index_pointer:index_pointer + len(images)]
                file_paths = [dataset.dataset.samples[i][0] for i in batch_indices]
                file_paths_list.extend(file_paths)
                index_pointer += len(images)
            else:
                file_paths_list.extend(['N/A'] * len(labels))

    df = pd.DataFrame({
        'file_path': file_paths_list,
        'label': labels_list,
        'embedding': embeddings_list,
        'augmented_image': augmented_images_list
    })
    return df

def display_all_augmented_images(df, class_names, max_images=None):
    """Exibe imagens augmentadas."""
    if max_images is not None:
        df = df.head(max_images)
        st.write(f"**Visualização das Primeiras {max_images} Imagens**")
    else:
        st.write("**Visualização de Todas as Imagens**")
    num_images = len(df)
    if num_images == 0:
        st.write("Nenhuma imagem para exibir.")
        return
    cols_per_row = 5
    rows = (num_images + cols_per_row - 1) // cols_per_row
    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col in range(cols_per_row):
            idx = row * cols_per_row + col
            if idx < num_images:
                image = df.iloc[idx]['augmented_image']
                label = df.iloc[idx]['label']
                with cols[col]:
                    st.image(image, caption=class_names[label], use_column_width=True)

def visualize_embeddings(df, class_names):
    """PCA em embeddings."""
    embeddings = np.vstack(df['embedding'].values)
    labels = df['label'].values
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    plot_df = pd.DataFrame({
        'PC1': embeddings_2d[:, 0],
        'PC2': embeddings_2d[:, 1],
        'label': labels
    })
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='label', palette='Set2', legend='full')
    plt.title('Embeddings PCA')
    plt.legend(title='Classes', labels=class_names)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    st.pyplot(plt)
    plt.close()

def train_segmentation_model(images_dir, masks_dir, num_classes):
    """Treina modelo de segmentação básico (FCN-ResNet50)."""
    set_seed(42)
    batch_size = 4
    num_epochs = 25
    learning_rate = 0.001

    input_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    target_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = SegmentationDataset(images_dir, masks_dir, transform=input_transforms, target_transform=target_transforms)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    if train_size == 0 or val_size == 0:
        st.error("Dataset de segmentação muito pequeno.")
        return None
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)
    model = get_segmentation_model(num_classes=num_classes, fine_tune=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, masks in train_loader:
            inputs = inputs.to(device)
            masks = masks.to(device).long().squeeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        st.write(f"Época [{epoch+1}/{num_epochs}], Perda Treino: {epoch_loss:.4f}")
        # Valid
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs = inputs.to(device)
                masks = masks.to(device).long().squeeze(1)
                outputs = model(inputs)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        st.write(f"Época [{epoch+1}/{num_epochs}], Perda Validação: {val_loss:.4f}")
    return model

def train_model(data_dir, num_classes, model_name, fine_tune, epochs, learning_rate, batch_size, train_split, valid_split, use_weighted_loss, l2_lambda, patience):
    """
    Treinamento do modelo de classificação.
    """
    set_seed(42)
    full_dataset = datasets.ImageFolder(root=data_dir)
    if len(full_dataset.classes) < num_classes:
        st.error(f"Menos classes ({len(full_dataset.classes)}) do que o especificado ({num_classes}).")
        return None
    visualize_data(full_dataset, full_dataset.classes)
    plot_class_distribution(full_dataset, full_dataset.classes)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_end = int(train_split * dataset_size)
    valid_end = int((train_split + valid_split) * dataset_size)
    train_indices = indices[:train_end]
    valid_indices = indices[train_end:valid_end]
    test_indices = indices[valid_end:]
    if len(train_indices) == 0 or len(valid_indices) == 0 or len(test_indices) == 0:
        st.error("Divisão de dados resultou em algum conjunto vazio.")
        return None
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(full_dataset, valid_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # Modelo para extrair embeddings
    model_for_embeddings = get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False)
    if model_for_embeddings is None:
        return None

    st.write("**Processando Treino (Data Aug + Embeddings)**")
    train_df = apply_transforms_and_get_embeddings(train_dataset, model_for_embeddings, train_transforms, batch_size=batch_size)
    st.write("**Processando Validação**")
    valid_df = apply_transforms_and_get_embeddings(valid_dataset, model_for_embeddings, test_transforms, batch_size=batch_size)
    st.write("**Processando Teste**")
    test_df = apply_transforms_and_get_embeddings(test_dataset, model_for_embeddings, test_transforms, batch_size=batch_size)

    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    train_df['class_name'] = train_df['label'].map(idx_to_class)
    valid_df['class_name'] = valid_df['label'].map(idx_to_class)
    test_df['class_name'] = test_df['label'].map(idx_to_class)

    st.write("**DF Treino c/ Aug + Embeddings:**")
    st.dataframe(train_df.drop(columns=['augmented_image']))
    st.write("**DF Val:**")
    st.dataframe(valid_df.drop(columns=['augmented_image']))
    st.write("**DF Test:**")
    st.dataframe(test_df.drop(columns=['augmented_image']))

    display_all_augmented_images(train_df, full_dataset.classes, max_images=50)
    visualize_embeddings(train_df, full_dataset.classes)

    st.write("**Distribuição de Classes (Treino)**")
    train_class_counts = train_df['class_name'].value_counts()
    st.bar_chart(train_class_counts)
    st.write("**Distribuição de Classes (Teste)**")
    test_class_counts = test_df['class_name'].value_counts()
    st.bar_chart(test_class_counts)

    # Reconfigurar para DataLoader
    train_dataset = CustomDataset(torch.utils.data.Subset(full_dataset, train_indices), transform=train_transforms)
    valid_dataset = CustomDataset(torch.utils.data.Subset(full_dataset, valid_indices), transform=test_transforms)
    test_dataset = CustomDataset(torch.utils.data.Subset(full_dataset, test_indices), transform=test_transforms)

    g = torch.Generator()
    g.manual_seed(42)

    if use_weighted_loss:
        targets = [full_dataset.targets[i] for i in train_indices]
        class_counts = np.bincount(targets)
        class_counts = class_counts + 1e-6
        class_weights = 1.0 / class_counts
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

    # Modelo final
    model = get_model(model_name, num_classes, dropout_p=0.5, fine_tune=fine_tune)
    if model is None:
        return None

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=l2_lambda)
    if 'train_losses' not in st.session_state:
        st.session_state.train_losses = []
    if 'valid_losses' not in st.session_state:
        st.session_state.valid_losses = []
    if 'train_accuracies' not in st.session_state:
        st.session_state.train_accuracies = []
    if 'valid_accuracies' not in st.session_state:
        st.session_state.valid_accuracies = []

    best_valid_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None

    placeholder = st.empty()
    progress_bar = st.progress(0)
    epoch_text = st.empty()

    for epoch in range(epochs):
        set_seed(42 + epoch)
        running_loss = 0.0
        running_corrects = 0
        model.train()

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            try:
                outputs = model(inputs)
            except Exception as e:
                st.error(f"Erro durante treinamento: {e}")
                return None
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        st.session_state.train_losses.append(epoch_loss)
        st.session_state.train_accuracies.append(epoch_acc.item())

        model.eval()
        valid_running_loss = 0.0
        valid_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                valid_running_loss += loss.item() * inputs.size(0)
                valid_running_corrects += torch.sum(preds == labels.data)
        valid_epoch_loss = valid_running_loss / len(valid_dataset)
        valid_epoch_acc = valid_running_corrects.double() / len(valid_dataset)
        st.session_state.valid_losses.append(valid_epoch_loss)
        st.session_state.valid_accuracies.append(valid_epoch_acc.item())

        with placeholder.container():
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ax[0].plot(range(1, len(st.session_state.train_losses) + 1), st.session_state.train_losses, label='Treino')
            ax[0].plot(range(1, len(st.session_state.valid_losses) + 1), st.session_state.valid_losses, label='Validação')
            ax[0].set_title(f'Perda por Época ({timestamp})')
            ax[0].set_xlabel('Épocas')
            ax[0].set_ylabel('Perda')
            ax[0].legend()

            ax[1].plot(range(1, len(st.session_state.train_accuracies) + 1), st.session_state.train_accuracies, label='Treino')
            ax[1].plot(range(1, len(st.session_state.valid_accuracies) + 1), st.session_state.valid_accuracies, label='Validação')
            ax[1].set_title(f'Acurácia por Época ({timestamp})')
            ax[1].set_xlabel('Épocas')
            ax[1].set_ylabel('Acurácia')
            ax[1].legend()

            st.pyplot(fig)
            plt.close(fig)

        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        epoch_text.text(f'Época {epoch+1}/{epochs}')

        # EarlyStopping
        if valid_epoch_loss < best_valid_loss:
            best_valid_loss = valid_epoch_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                st.write("Early stopping!")
                if best_model_wts is not None:
                    model.load_state_dict(best_model_wts)
                break

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # Plotagem final
    plot_metrics(st.session_state.train_losses, st.session_state.valid_losses,
                 st.session_state.train_accuracies, st.session_state.valid_accuracies)

    st.write("**Avaliação no Conjunto de Teste**")
    compute_metrics(model, test_loader, full_dataset.classes)
    st.write("**Análise de Erros**")
    error_analysis(model, test_loader, full_dataset.classes)
    st.write("**Clusterização**")
    perform_clustering(model, test_loader, full_dataset.classes)

    del train_loader, valid_loader
    gc.collect()
    st.session_state['model'] = model
    st.session_state['classes'] = full_dataset.classes
    st.session_state['trained_model_name'] = model_name

    return model, full_dataset.classes

def plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies):
    """Gráfico final de Perda e Acurácia."""
    epochs_range = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ax[0].plot(epochs_range, train_losses, label='Treino')
    ax[0].plot(epochs_range, valid_losses, label='Validação')
    ax[0].set_title(f'Perda por Época ({timestamp})')
    ax[0].set_xlabel('Épocas')
    ax[0].set_ylabel('Perda')
    ax[0].legend()

    ax[1].plot(epochs_range, train_accuracies, label='Treino')
    ax[1].plot(epochs_range, valid_accuracies, label='Validação')
    ax[1].set_title(f'Acurácia por Época ({timestamp})')
    ax[1].set_xlabel('Épocas')
    ax[1].set_ylabel('Acurácia')
    ax[1].legend()

    st.pyplot(fig)
    plt.close(fig)

def compute_metrics(model, dataloader, classes):
    """Calcula métricas, imprime matriz de confusão e classification_report."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    st.text("Relatório de Classificação:")
    st.write(pd.DataFrame(report).transpose())

    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    ax.set_title('Matriz de Confusão (Normalizada)')
    st.pyplot(fig)
    plt.close(fig)

    if len(classes) == 2:
        fpr, tpr, thresholds = roc_curve(all_labels, [p[1] for p in all_probs])
        roc_auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
        ax.plot([0,1],[0,1],'k--')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('Curva ROC (Binária)')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    else:
        binarized_labels = label_binarize(all_labels, classes=range(len(classes)))
        roc_auc = roc_auc_score(binarized_labels, np.array(all_probs), average='weighted', multi_class='ovr')
        st.write(f"AUC-ROC Média Ponderada (multiclasse): {roc_auc:.4f}")

def error_analysis(model, dataloader, classes):
    """Exibir imagens mal classificadas."""
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            incorrect = preds != labels
            if incorrect.any():
                misclassified_images.extend(inputs[incorrect].cpu())
                misclassified_labels.extend(labels[incorrect].cpu())
                misclassified_preds.extend(preds[incorrect].cpu())
                if len(misclassified_images) >= 5:
                    break
    if misclassified_images:
        st.write("Imagens mal classificadas:")
        fig, axes = plt.subplots(1, min(5, len(misclassified_images)), figsize=(15,3))
        for i in range(min(5, len(misclassified_images))):
            image = misclassified_images[i].permute(1,2,0).numpy()
            axes[i].imshow(image)
            axes[i].set_title(f"V: {classes[misclassified_labels[i]]}\nP: {classes[misclassified_preds[i]]}")
            axes[i].axis('off')
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.write("Nenhuma imagem mal classificada encontrada.")

def perform_clustering(model, dataloader, classes):
    """Extração de features + clustering."""
    features = []
    labels = []
    if isinstance(model, nn.Sequential):
        model_feat = model
    else:
        model_feat = nn.Sequential(*list(model.children())[:-1])
    model_feat.eval()
    model_feat.to(device)

    with torch.no_grad():
        for inputs, lbl in dataloader:
            inputs = inputs.to(device)
            out = model_feat(inputs)
            out = out.view(out.size(0), -1)
            features.append(out.cpu().numpy())
            labels.extend(lbl.numpy())
    features = np.vstack(features)
    labels = np.array(labels)

    pca = PCA(n_components=2)
    feats_2d = pca.fit_transform(features)

    kmeans = KMeans(n_clusters=len(classes), random_state=42)
    clusters_km = kmeans.fit_predict(features)

    agglo = AgglomerativeClustering(n_clusters=len(classes))
    clusters_ag = agglo.fit_predict(features)

    fig, ax = plt.subplots(1,2, figsize=(14,6))

    sc = ax[0].scatter(feats_2d[:,0], feats_2d[:,1], c=clusters_km, cmap='viridis')
    ax[0].legend(*sc.legend_elements(), title="Clusters")
    ax[0].set_title("K-Means")

    sc = ax[1].scatter(feats_2d[:,0], feats_2d[:,1], c=clusters_ag, cmap='viridis')
    ax[1].legend(*sc.legend_elements(), title="Clusters")
    ax[1].set_title("Hierárquico")

    st.pyplot(fig)
    plt.close(fig)

    ari_km = adjusted_rand_score(labels, clusters_km)
    nmi_km = normalized_mutual_info_score(labels, clusters_km)
    ari_ag = adjusted_rand_score(labels, clusters_ag)
    nmi_ag = normalized_mutual_info_score(labels, clusters_ag)

    st.write(f"K-Means ARI={ari_km:.4f}, NMI={nmi_km:.4f}")
    st.write(f"Agglo ARI={ari_ag:.4f}, NMI={nmi_ag:.4f}")

def evaluate_image(model, image, classes):
    """Faz uma predição para imagem individual."""
    model.eval()
    image_tensor = test_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        idx = pred.item()
        cname = classes[idx]
        return cname, conf.item()

def label_to_color_image(label):
    """Mapeia label de segmentação para imagem colorida."""
    colormap = create_pascal_label_colormap()
    return colormap[label]

def create_pascal_label_colormap():
    """Colormap Pascal VOC."""
    colormap = np.zeros((256,3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:,channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap

def visualize_activations(model, image, class_names, model_name, segmentation_model=None, segmentation=False):
    """Visualização com Grad-CAM + Segmentação opcional."""
    model.eval()
    input_tensor = test_transforms(image).unsqueeze(0).to(device)

    # Indica a camada-alvo para ResNet ou DenseNet
    if model_name.startswith('ResNet'):
        target_layer = 'layer4'
    elif model_name.startswith('DenseNet'):
        target_layer = 'features.denseblock4'
    else:
        st.error("Modelo não suportado para Grad-CAM.")
        return

    cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer)

    with torch.set_grad_enabled(True):
        out = model(input_tensor)
        probs = torch.nn.functional.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
        pred_class = pred.item()
        activation_map = cam_extractor(pred_class, out)

    activation_map = activation_map[0]
    result = overlay_mask(
        to_pil_image(input_tensor.squeeze().cpu()),
        to_pil_image(activation_map.squeeze(), mode='F'),
        alpha=0.5
    )

    image_np = np.array(image)

    if segmentation and segmentation_model is not None:
        segmentation_model.eval()
        with torch.no_grad():
            seg_output = segmentation_model(input_tensor)['out']
            seg_mask = torch.argmax(seg_output.squeeze(), dim=0).cpu().numpy()
        seg_colored = label_to_color_image(seg_mask).astype(np.uint8)
        seg_colored = cv2.resize(seg_colored, (image.size[0], image.size[1]))

        fig, ax = plt.subplots(1, 3, figsize=(15,5))
        ax[0].imshow(image_np)
        ax[0].set_title("Original")
        ax[0].axis('off')

        ax[1].imshow(result)
        ax[1].set_title("Grad-CAM")
        ax[1].axis('off')

        ax[2].imshow(image_np)
        ax[2].imshow(seg_colored, alpha=0.6)
        ax[2].set_title("Segmentação")
        ax[2].axis('off')

        st.pyplot(fig)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        ax[0].imshow(image_np)
        ax[0].set_title("Imagem Original")
        ax[0].axis('off')
        ax[1].imshow(result)
        ax[1].set_title("Grad-CAM")
        ax[1].axis('off')
        st.pyplot(fig)
        plt.close(fig)

def main():
    icon_path = "logo.png"
    if os.path.exists(icon_path):
        try:
            st.set_page_config(page_title="Geomaker", page_icon=icon_path, layout="wide")
            logging.info(f"Ícone {icon_path} carregado.")
        except Exception as e:
            st.set_page_config(page_title="Geomaker", layout="wide")
    else:
        st.set_page_config(page_title="Geomaker", layout="wide")

    if os.path.exists('capa.png'):
        try:
            st.image('capa.png', width=100,
                     caption='Laboratório de Educação e IA - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay',
                     use_container_width=True)
        except UnidentifiedImageError:
            st.warning("capa.png corrompida.")
    else:
        st.warning("capa.png não encontrada.")

    if os.path.exists("logo.png"):
        try:
            st.sidebar.image("logo.png", width=200)
        except UnidentifiedImageError:
            st.sidebar.text("logo.png corrompido.")
    else:
        st.sidebar.text("logo.png não encontrado.")

    st.title("Classificação e Segmentação de Imagens + Grad-CAM (SmoothGradCAMpp)")
    st.write("Exemplo de aplicativo Streamlit com PyTorch, Data Aug, Grad-CAM e Segmentação.")
    st.write("Caso o `torchcam` não esteja instalado, execute: `pip install torch-cam`")

    # Decidir se iremos usar segmentação
    st.subheader("Opções de Segmentação")
    segmentation_option = st.selectbox("Deseja utilizar Segmentação?", ["Não", "Pré-treinado FCN ResNet50", "Treinar um novo para segmentação"])
    segmentation_model = None
    if segmentation_option == "Pré-treinado FCN ResNet50":
        num_classes_seg = st.number_input("Número de classes (FCN)", min_value=1, value=21)
        segmentation_model = get_segmentation_model(num_classes_seg)
        st.write("Modelo de segmentação pré-treinado OK.")
    elif segmentation_option == "Treinar um novo para segmentação":
        num_classes_seg = st.number_input("Número classes segmentação:", min_value=1, value=2)
        seg_zip = st.file_uploader("Zip com pasta images/ e masks/", type=["zip"])
        if seg_zip:
            temp_seg_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_seg_dir, "segmentation.zip")
            with open(zip_path, "wb") as f:
                f.write(seg_zip.read())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_seg_dir)
            images_dir = os.path.join(temp_seg_dir, 'images')
            masks_dir = os.path.join(temp_seg_dir, 'masks')
            if os.path.exists(images_dir) and os.path.exists(masks_dir):
                st.write("Treinando modelo de segmentação..")
                segmentation_model = train_segmentation_model(images_dir, masks_dir, num_classes_seg)
                if segmentation_model is not None:
                    st.success("Segmentação treinada.")
            else:
                st.error("Estrutura de pastas não encontrada (images/, masks/).")

    # Configs de Classificação
    st.sidebar.title("Configurações de Classificação")
    num_classes = st.sidebar.number_input("Nº classes", min_value=2, value=2)
    model_name = st.sidebar.selectbox("Modelo:", ["ResNet18","ResNet50","DenseNet121"])
    fine_tune = st.sidebar.checkbox("Fine Tune", value=False)
    epochs = st.sidebar.slider("Épocas", min_value=1, max_value=200, value=10)
    learning_rate = st.sidebar.select_slider("LR:", options=[0.1, 0.01, 0.001, 0.0001], value=0.001)
    batch_size = st.sidebar.selectbox("Batch Size", [4,8,16,32], index=2)
    train_split = st.sidebar.slider("Treino (%)", 0.5, 0.9, 0.7, 0.05)
    valid_split = st.sidebar.slider("Val (%)", 0.05, 0.4, 0.15, 0.05)
    l2_lambda = st.sidebar.number_input("Weight Decay (L2):", min_value=0.0, max_value=0.1, value=0.01, step=0.01)
    patience = st.sidebar.number_input("EarlyStopping Paciencia:", min_value=1, max_value=10, value=3)
    use_weighted_loss = st.sidebar.checkbox("Loss Ponderada p/ Desbalanceamento?", value=False)

    st.sidebar.write("""
    By [Geomaker+IA](https://doi.org/10.5281/zenodo.13910277).  
    - Prof. Marcelo Claro  
    - Contatos: marceloclaro@gmail.com
    """)

    model_option = st.selectbox("Carregar / Treinar?", ["Treinar Novo Modelo", "Carregar Modelo Existente"])
    if model_option == "Carregar Modelo Existente":
        loaded_model_file = st.file_uploader("Modelo .pth", type=["pth","pt"])
        if loaded_model_file is not None:
            model_loaded = get_model(model_name, num_classes, 0.5, fine_tune=False)
            if model_loaded is None:
                st.error("Erro ao criar arch do modelo.")
                return
            try:
                loaded_sd = torch.load(loaded_model_file, map_location=device)
                model_loaded.load_state_dict(loaded_sd)
                st.session_state['model'] = model_loaded
                st.success("Modelo carregado!")
            except Exception as e:
                st.error(f"Falha ao load: {e}")

        classes_file = st.file_uploader("classes.txt", type=["txt"])
        if classes_file:
            cls_text = classes_file.read().decode("utf-8").splitlines()
            st.session_state['classes'] = cls_text
            st.write(f"Classes: {cls_text}")
    else:
        # Treino do zero
        zip_file = st.file_uploader("ZIP com as imagens (pastas=classes)", type=["zip"])
        if zip_file and train_split+valid_split <=0.95:
            tmp_dir = tempfile.mkdtemp()
            zpath = os.path.join(tmp_dir, "images.zip")
            with open(zpath, "wb") as f:
                f.write(zip_file.read())
            with zipfile.ZipFile(zpath,'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
            data_dir = tmp_dir
            st.write("Treinando..")
            train_res = train_model(
                data_dir, num_classes, model_name, fine_tune, epochs, learning_rate, batch_size,
                train_split, valid_split, use_weighted_loss, l2_lambda, patience
            )
            if train_res:
                model_tr, classes_tr = train_res
                st.success("Treino concluído!")
                # Download do modelo
                buf = io.BytesIO()
                torch.save(model_tr.state_dict(), buf)
                buf.seek(0)
                st.download_button("Baixar Modelo Treinado", data=buf.getvalue(), file_name="modelo_treinado.pth", mime="application/octet-stream")
                # Baixar classes
                cls_str = "\n".join(classes_tr)
                st.download_button("Baixar Classes", data=cls_str, file_name="classes.txt", mime="text/plain")

            # Limpeza
            shutil.rmtree(tmp_dir)
        else:
            st.info("Aguardando ZIP ou splits adequados.")

    # Avaliar imagem
    st.header("Avaliar Imagem")
    eval_choice = st.radio("Deseja avaliar?", ["Sim","Não"])
    if eval_choice=="Sim":
        if 'model' not in st.session_state or 'classes' not in st.session_state:
            st.warning("Sem modelo ou classes.")
        else:
            up_img_eval = st.file_uploader("Imagem", type=["png","jpg","jpeg","bmp"])
            if up_img_eval:
                up_img_eval.seek(0)
                try:
                    eval_im = Image.open(up_img_eval).convert("RGB")
                    st.image(eval_im, caption='Imagem Avaliar', use_column_width=True)
                    pred_cl, conf_ = evaluate_image(st.session_state['model'], eval_im, st.session_state['classes'])
                    st.write(f"Classe Predita: {pred_cl}, Confiança={conf_:.4f}")
                    # Visualizar Grad-CAM
                    seg_checkbox = st.checkbox("Visualizar Segmentação (se disponível)?", value=False)
                    seg_to_use = segmentation_model if seg_checkbox else None
                    nm = st.session_state.get('trained_model_name', model_name)
                    visualize_activations(st.session_state['model'], eval_im, st.session_state['classes'], nm, segmentation_model=seg_to_use, segmentation=seg_checkbox)
                except Exception as e:
                    st.error(f"Erro ao abrir: {e}")
    st.write("**Fim da execução**")

if __name__ == "__main__":
    main()

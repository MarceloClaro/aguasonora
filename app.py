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

# Importar SOMENTE os modelos que funcionam em PyTorch + Torchvision em Python 3.12
from torchvision.models import (
    resnet18,
    resnet50,
    densenet121,
    # Removendo quaisquer imports de fcn_resnet50 ou segmentação
)

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA

import streamlit as st
import gc
import logging
import base64
import io
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.classes.*")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sns.set_style('whitegrid')

def set_seed(seed):
    """
    Define uma seed para garantir a reprodutibilidade.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Fixa seed para efeitos de demo

################################################################################
#                               DATA TRANSFORMS
################################################################################

# Aumento de dados no treinamento
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Apenas resize e crop para validação e teste
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

################################################################################
#                           DATASET CUSTOM
################################################################################

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

def seed_worker(worker_id):
    """
    Define seed no worker do DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

################################################################################
#                         VISUALIZAÇÃO DO DATASET
################################################################################

def visualize_data(dataset, classes):
    """
    Mostra algumas imagens do dataset, com suas classes.
    """
    st.write("Visualização de algumas imagens do dataset:")
    fig, axes = plt.subplots(1, min(10, len(dataset)), figsize=(20, 4))
    for i in range(min(10, len(dataset))):
        idx = np.random.randint(len(dataset))
        image, label = dataset[idx]
        image = np.array(image)
        axes[i].imshow(image)
        axes[i].set_title(classes[label])
        axes[i].axis('off')
    st.pyplot(fig)
    plt.close(fig)

def plot_class_distribution(dataset, classes):
    """
    Mostra distribuição das classes (contagem) no dataset.
    """
    labels = [label for _, label in dataset]
    df = pd.DataFrame({'Classe': labels})
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Classe', data=df, palette="Set2", ax=ax)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_title("Distribuição das Classes")
    for i, count in enumerate(df['Classe'].value_counts().sort_index()):
        ax.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    st.pyplot(fig)
    plt.close(fig)

################################################################################
#                              GET MODEL
################################################################################

def get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False):
    """
    Retorna ResNet ou DenseNet, ajustando a última camada para num_classes.
    """
    if model_name == 'ResNet18':
        model = resnet18(weights="IMAGENET1K_V1")
    elif model_name == 'ResNet50':
        model = resnet50(weights="IMAGENET1K_V1")
    elif model_name == 'DenseNet121':
        model = densenet121(weights="IMAGENET1K_V1")
    else:
        st.error("Modelo não suportado.")
        return None

    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    # Ajusta a última camada
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

    return model.to(device)

################################################################################
#                          APPLY TRANSFORMS & EMBEDDINGS
################################################################################

def apply_transforms_and_get_embeddings(dataset, model, transform, batch_size=16):
    """
    Extrai embeddings (removendo última camada do modelo) e retorna DataFrame.
    """
    def pil_collate_fn(batch):
        images, labels = zip(*batch)
        return list(images), torch.tensor(labels)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=pil_collate_fn)
    embeddings_list = []
    labels_list = []
    file_paths_list = []
    augmented_images_list = []

    # Remove última camada (classificador)
    model_embedding = nn.Sequential(*list(model.children())[:-1])
    model_embedding.eval()
    model_embedding.to(device)

    indices = dataset.indices if hasattr(dataset, 'indices') else list(range(len(dataset)))
    index_pointer = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images_aug = [transform(img) for img in images]
            images_aug = torch.stack(images_aug).to(device)
            emb = model_embedding(images_aug)
            emb = emb.view(emb.size(0), -1).cpu().numpy()
            embeddings_list.extend(emb)
            labels_list.extend(labels.numpy())
            augmented_images_list.extend([img.permute(1, 2, 0).numpy() for img in images_aug.cpu()])

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

def display_all_augmented_images(df, class_names, max_images=50):
    """
    Exibe imagens após Data Augmentation armazenadas no df['augmented_image'].
    """
    df_show = df.head(max_images)
    num_images = len(df_show)
    st.write(f"**Visualização das Primeiras {num_images} Imagens após Data Augmentation:**")

    cols_per_row = 5
    rows = (num_images + cols_per_row - 1) // cols_per_row
    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col in range(cols_per_row):
            idx = row * cols_per_row + col
            if idx < num_images:
                image_np = df_show.iloc[idx]['augmented_image']
                label = df_show.iloc[idx]['label']
                with cols[col]:
                    st.image(image_np, caption=class_names[label], use_column_width=True)

def visualize_embeddings(df, class_names):
    """
    Aplica PCA nos embeddings e plota em 2D.
    """
    embeddings = np.vstack(df['embedding'].values)
    labels = df['label'].values

    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)

    plot_df = pd.DataFrame({
        'PC1': emb_2d[:,0],
        'PC2': emb_2d[:,1],
        'label': labels
    })
    fig = plt.figure(figsize=(10,7))
    sns.scatterplot(data=plot_df, x='PC1', y='PC2',
                    hue='label', palette='Set2', legend='full')
    plt.title("PCA dos Embeddings")
    plt.legend(title='Classes', labels=class_names)
    st.pyplot(fig)
    plt.close(fig)

################################################################################
#                          TREINAR MODELO
################################################################################

def train_model(data_dir, num_classes, model_name, fine_tune, epochs,
                learning_rate, batch_size, train_split, valid_split,
                use_weighted_loss, l2_lambda, patience):
    """
    Função principal: Treina o modelo de classificação.
    """
    set_seed(42)
    full_dataset = datasets.ImageFolder(root=data_dir)
    if len(full_dataset.classes) < num_classes:
        st.error(f"N. de classes encontradas ({len(full_dataset.classes)}) < n. especificado ({num_classes}).")
        return None

    visualize_data(full_dataset, full_dataset.classes)
    plot_class_distribution(full_dataset, full_dataset.classes)

    # Split
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_end = int(train_split * dataset_size)
    valid_end = int((train_split + valid_split) * dataset_size)

    train_indices = indices[:train_end]
    valid_indices = indices[train_end:valid_end]
    test_indices  = indices[valid_end:]

    if len(train_indices)==0 or len(valid_indices)==0 or len(test_indices)==0:
        st.error("Conjunto vazio. Ajuste os splits.")
        return None

    train_sub = torch.utils.data.Subset(full_dataset, train_indices)
    valid_sub = torch.utils.data.Subset(full_dataset, valid_indices)
    test_sub  = torch.utils.data.Subset(full_dataset, test_indices)

    # Model p/ extrair embeddings
    model_for_emb = get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False)
    if model_for_emb is None: return None

    st.write("**Processando Treino (Data Aug + Embeddings)...**")
    train_df = apply_transforms_and_get_embeddings(train_sub, model_for_emb, train_transforms, batch_size)
    st.write("**Processando Validação...**")
    valid_df = apply_transforms_and_get_embeddings(valid_sub, model_for_emb, test_transforms, batch_size)
    st.write("**Processando Teste...**")
    test_df  = apply_transforms_and_get_embeddings(test_sub,  model_for_emb, test_transforms, batch_size)

    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    train_df['class_name'] = train_df['label'].map(idx_to_class)
    valid_df['class_name'] = valid_df['label'].map(idx_to_class)
    test_df['class_name']  = test_df['label'].map(idx_to_class)

    st.write("**Dataframe Train (sem col. image):**")
    st.dataframe(train_df.drop(columns=['augmented_image']))
    st.write("**Dataframe Val:**")
    st.dataframe(valid_df.drop(columns=['augmented_image']))
    st.write("**Dataframe Test:**")
    st.dataframe(test_df.drop(columns=['augmented_image']))

    # Imagens
    display_all_augmented_images(train_df, full_dataset.classes, max_images=30)
    visualize_embeddings(train_df, full_dataset.classes)

    # Dist. classes
    st.write("**Dist. Classes Treino:**")
    st.bar_chart(train_df['class_name'].value_counts())

    st.write("**Dist. Classes Teste:**")
    st.bar_chart(test_df['class_name'].value_counts())

    # Final dataset c/ transforms
    train_ds = CustomDataset(train_sub, transform=train_transforms)
    valid_ds = CustomDataset(valid_sub, transform=test_transforms)
    test_ds  = CustomDataset(test_sub,  transform=test_transforms)

    # Weighted loss?
    if use_weighted_loss:
        targets = [full_dataset.targets[i] for i in train_indices]
        cc = np.bincount(targets) + 1e-6
        cw = 1.0 / cc
        cw = torch.FloatTensor(cw).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              worker_init_fn=seed_worker, generator=g)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              worker_init_fn=seed_worker, generator=g)

    # Modelo final
    model = get_model(model_name, num_classes, dropout_p=0.5, fine_tune=fine_tune)
    if model is None: return None
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=learning_rate, weight_decay=l2_lambda)

    # Variáveis globais p/ gráfico
    if 'train_losses' not in st.session_state: st.session_state.train_losses = []
    if 'valid_losses' not in st.session_state: st.session_state.valid_losses = []
    if 'train_accuracies' not in st.session_state: st.session_state.train_accuracies = []
    if 'valid_accuracies' not in st.session_state: st.session_state.valid_accuracies = []

    best_valid_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None

    placeholder = st.empty()
    pb = st.progress(0)
    epoch_txt = st.empty()

    for ep in range(epochs):
        set_seed(42 + ep)
        running_loss = 0.0
        running_corrects = 0
        model.train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(inputs)
            _, preds = torch.max(out, 1)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_ds)
        epoch_acc = running_corrects.double() / len(train_ds)
        st.session_state.train_losses.append(epoch_loss)
        st.session_state.train_accuracies.append(epoch_acc.item())

        model.eval()
        valid_loss = 0.0
        valid_corrects = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                out = model(inputs)
                _, preds = torch.max(out, 1)
                l = criterion(out, labels)
                valid_loss += l.item() * inputs.size(0)
                valid_corrects += torch.sum(preds == labels.data)
        v_loss = valid_loss / len(valid_ds)
        v_acc  = valid_corrects.double() / len(valid_ds)
        st.session_state.valid_losses.append(v_loss)
        st.session_state.valid_accuracies.append(v_acc.item())

        with placeholder.container():
            fig, ax = plt.subplots(1,2, figsize=(14,5))
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            ax[0].plot(range(1, len(st.session_state.train_losses)+1),
                       st.session_state.train_losses, label='Treino')
            ax[0].plot(range(1, len(st.session_state.valid_losses)+1),
                       st.session_state.valid_losses, label='Val')
            ax[0].set_title(f'Perda (at {ts})')
            ax[0].legend()

            ax[1].plot(range(1, len(st.session_state.train_accuracies)+1),
                       st.session_state.train_accuracies, label='Treino')
            ax[1].plot(range(1, len(st.session_state.valid_accuracies)+1),
                       st.session_state.valid_accuracies, label='Val')
            ax[1].set_title(f'Acurácia (at {ts})')
            ax[1].legend()

            st.pyplot(fig)
            plt.close(fig)

        pr = (ep+1)/epochs
        pb.progress(pr)
        epoch_txt.text(f'Época {ep+1}/{epochs}')

        # Early stopping
        if v_loss < best_valid_loss:
            best_valid_loss = v_loss
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

    # Plot final
    plot_metrics(st.session_state.train_losses, st.session_state.valid_losses,
                 st.session_state.train_accuracies, st.session_state.valid_accuracies)

    st.write("**Avaliação no Conjunto de Teste**")
    compute_metrics(model, test_loader, full_dataset.classes)
    st.write("**Análise de Erros**")
    error_analysis(model, test_loader, full_dataset.classes)
    st.write("**Análise de Cluster**")
    perform_clustering(model, test_loader, full_dataset.classes)

    # Limpando
    del train_loader, valid_loader, test_loader
    gc.collect()
    st.session_state['model'] = model
    st.session_state['classes'] = full_dataset.classes
    return model, full_dataset.classes

################################################################################
#                        PLOT FINAL & MÉTRICAS
################################################################################

def plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies):
    ep_range = range(1, len(train_losses)+1)
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ax[0].plot(ep_range, train_losses, label='Treino')
    ax[0].plot(ep_range, valid_losses, label='Val')
    ax[0].set_title(f'Perda (final) - {ts}')
    ax[0].legend()

    ax[1].plot(ep_range, train_accuracies, label='Treino')
    ax[1].plot(ep_range, valid_accuracies, label='Val')
    ax[1].set_title(f'Acurácia (final) - {ts}')
    ax[1].legend()

    st.pyplot(fig)
    plt.close(fig)

def compute_metrics(model, dataloader, classes):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            out = model(inputs)
            probs = torch.nn.functional.softmax(out, dim=1)
            _, preds = torch.max(out, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Relatório
    rep = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    st.write(pd.DataFrame(rep).transpose())

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title("Matriz de Confusão Normalizada")
    st.pyplot(fig)
    plt.close(fig)

    # ROC
    if len(classes)==2:
        fpr, tpr, _ = roc_curve(all_labels, [p[1] for p in all_probs])
        aucv = roc_auc_score(all_labels, [p[1] for p in all_probs])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={aucv:.2f}")
        ax.plot([0,1],[0,1], 'k--')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    else:
        bin_labels = label_binarize(all_labels, classes=range(len(classes)))
        aucv = roc_auc_score(bin_labels, np.array(all_probs),
                             average='weighted', multi_class='ovr')
        st.write(f"AUC-ROC (média ponderada): {aucv:.4f}")

def error_analysis(model, dataloader, classes):
    model.eval()
    mis_images = []
    mis_labels = []
    mis_preds  = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            _, preds = torch.max(out,1)
            incorrect = (preds!=labels)
            if incorrect.any():
                mis_images.extend(inputs[incorrect].cpu())
                mis_labels.extend(labels[incorrect].cpu())
                mis_preds.extend(preds[incorrect].cpu())
                if len(mis_images)>=5:
                    break
    if mis_images:
        st.write("Exemplos de Erros de Classificação:")
        fig, axes = plt.subplots(1, min(5,len(mis_images)), figsize=(15,3))
        for i in range(min(5, len(mis_images))):
            image = mis_images[i]
            image = image.permute(1,2,0).numpy()
            axes[i].imshow(image)
            axes[i].set_title(f"V:{classes[mis_labels[i]]}\nP:{classes[mis_preds[i]]}")
            axes[i].axis('off')
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.write("Nenhum erro encontrado.")

def perform_clustering(model, dataloader, classes):
    # Extrai features
    features=[]
    labels=[]
    # Remove última camada
    if isinstance(model, nn.Sequential):
        model_feat = model
    else:
        model_feat = nn.Sequential(*list(model.children())[:-1])
    model_feat.eval()
    model_feat.to(device)

    with torch.no_grad():
        for inputs, label in dataloader:
            inputs = inputs.to(device)
            out = model_feat(inputs)
            out = out.view(out.size(0), -1)
            features.append(out.cpu().numpy())
            labels.extend(label.numpy())

    features = np.vstack(features)
    labels   = np.array(labels)

    # PCA
    pca = PCA(n_components=2)
    feat_2d = pca.fit_transform(features)

    # KMeans
    km = KMeans(n_clusters=len(classes), random_state=42)
    c_km = km.fit_predict(features)

    # Agglo
    agglo = AgglomerativeClustering(n_clusters=len(classes))
    c_ag = agglo.fit_predict(features)

    fig, ax = plt.subplots(1,2, figsize=(14,6))
    sc1 = ax[0].scatter(feat_2d[:,0], feat_2d[:,1], c=c_km, cmap='viridis')
    ax[0].set_title("KMeans")

    sc2 = ax[1].scatter(feat_2d[:,0], feat_2d[:,1], c=c_ag, cmap='viridis')
    ax[1].set_title("AgglomerativeClustering")
    st.pyplot(fig)
    plt.close(fig)

    ari_km  = adjusted_rand_score(labels, c_km)
    nmi_km  = normalized_mutual_info_score(labels, c_km)
    ari_ag  = adjusted_rand_score(labels, c_ag)
    nmi_ag  = normalized_mutual_info_score(labels, c_ag)

    st.write(f"KMeans => ARI:{ari_km:.4f}, NMI:{nmi_km:.4f}")
    st.write(f"Agglo => ARI:{ari_ag:.4f}, NMI:{nmi_ag:.4f}")

################################################################################
#                                    MAIN
################################################################################

def main():
    # Ajustar layout
    st.set_page_config(page_title="Classificador Imagens", layout="wide")

    st.title("Classificação de Imagens (PyTorch + Streamlit)")
    st.write("Exemplo mínimo de aplicação de classificação, sem segmentação.")

    st.sidebar.header("Configurações de Treinamento")
    num_classes = st.sidebar.number_input("Número de Classes:", min_value=2, step=1, value=2)
    model_name  = st.sidebar.selectbox("Modelo:", ["ResNet18", "ResNet50", "DenseNet121"], index=0)
    fine_tune   = st.sidebar.checkbox("Fine-Tune?", value=False)
    epochs      = st.sidebar.slider("Épocas:", 1, 50, 10)
    learning_rate = st.sidebar.select_slider("Taxa Aprendizagem", options=[1e-2,1e-3,1e-4], value=1e-3)
    batch_size  = st.sidebar.selectbox("Batch Size:", [4,8,16,32], index=1)
    train_split = st.sidebar.slider("Treino %",0.5,0.9,0.7,0.05)
    valid_split = st.sidebar.slider("Val %",0.05,0.4,0.15,0.05)
    l2_lambda   = st.sidebar.number_input("L2 (weight_decay):",0.0,0.1,0.01,0.01)
    patience    = st.sidebar.number_input("Paciencia EarlyStopping:",1,10,3,1)
    use_weighted_loss = st.sidebar.checkbox("Loss Ponderada (Desbalance)", value=False)

    st.sidebar.write("---")
    st.sidebar.write("Feito por: Geomaker + IA")

    if train_split + valid_split>0.95:
        st.sidebar.error("Soma Treino + Val > 0.95. Ajuste.")
        return

    # Upload dataset .zip
    st.subheader("Upload do Dataset Zip (pastas = classes)")
    zip_file = st.file_uploader("Selecione o .zip", type=["zip"])

    if zip_file is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "data.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_file.read())
            with zipfile.ZipFile(zip_path,'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            data_dir = tmpdir
            # Pastas => classes
            st.write("Treinando modelo ...")
            model_data = train_model(data_dir, num_classes, model_name, fine_tune,
                                     epochs, learning_rate, batch_size,
                                     train_split, valid_split,
                                     use_weighted_loss, l2_lambda, patience)

            if model_data is not None:
                model_final, classes_final = model_data
                st.success("Treinamento concluído!")
                # Opção de download
                with io.BytesIO() as buf:
                    torch.save(model_final.state_dict(), buf)
                    buf.seek(0)
                    st.download_button("Download Modelo Treinado",
                                       data=buf.getvalue(),
                                       file_name="modelo_treinado.pth")

                classes_str = "\n".join(classes_final)
                st.download_button("Download Classes",
                                   data=classes_str.encode("utf-8"),
                                   file_name="classes.txt")

    else:
        st.info("Aguardando upload do dataset .zip.")


if __name__ == "__main__":
    main()

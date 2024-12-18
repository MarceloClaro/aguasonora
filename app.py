import os
import zipfile
import tempfile
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

try:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from torchvision import transforms, models
    from sklearn.metrics import confusion_matrix, classification_report
except ImportError as e:
    print(f"Erro ao importar bibliotecas: {e}. Por favor, certifique-se de que todos os pacotes necessários estejam instalados.")
    raise

# Definir o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    """Define uma seed para garantir a reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Função para converter áudio em imagens (espectrogramas)
def audio_to_spectrogram(audio_path, output_dir, sr=22050, img_size=(224, 224)):
    try:
        import librosa
        import librosa.display

        if not audio_path.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            print(f"Arquivo ignorado (não é um áudio): {audio_path}")
            return

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
    except ImportError as e:
        print("O pacote 'librosa' não está instalado. Por favor, instale-o para processar áudios.")
        raise
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

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Erro ao abrir imagem {image_path}: {e}")
            raise

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

    # Verificar se o diretório de imagens existe
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {image_dir}")

    # Listar imagens e classes
    classes = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
    if not classes:
        raise ValueError("Nenhuma classe encontrada no diretório de imagens.")

    image_paths = []
    labels = []
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(image_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            if os.path.isfile(image_path):
                image_paths.append(image_path)
                labels.append(label_idx)

    if not image_paths:
        raise ValueError("Nenhuma imagem encontrada no diretório de imagens.")

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
    print("Classificação de Áudio usando Espectrogramas")

    # Configurações
    model_name = 'ResNet18'
    num_classes = 5
    epochs = 10
    learning_rate = 0.001
    batch_size = 16
    train_split = 0.7
    valid_split = 0.15

    # Caminho para o diretório de áudios
    audio_dir = "audios"
    spectrogram_dir = "spectrograms"

    if not os.path.exists(audio_dir):
        print(f"O diretório {audio_dir} não foi encontrado. Criando uma estrutura de exemplo...")
        os.makedirs(audio_dir, exist_ok=True)
        sample_class = os.path.join(audio_dir, "classe_exemplo")
        os.makedirs(sample_class, exist_ok=True)
        print(f"Por favor, adicione arquivos de áudio no diretório: {sample_class}")
        return

    os.makedirs(spectrogram_dir, exist_ok=True)

    # Converter áudios em espectrogramas
    for class_name in os.listdir(audio_dir):
        class_path = os.path.join(audio_dir, class_name)
        if not os.path.isdir(class_path):
            print(f"Ignorando {class_path}, pois não é um diretório.")
            continue

        output_class_dir = os.path.join(spectrogram_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for audio_file in os.listdir(class_path):
            audio_file_path = os.path.join(class_path, audio_file)
            try:
                audio_to_spectrogram(audio_file_path, output_class_dir)
            except Exception as e:
                print(f"Erro ao converter {audio_file}: {e}")

    # Verificar se há imagens no diretório de espectrogramas
    if not any(os.scandir(spectrogram_dir)):
        print(f"Nenhuma imagem foi gerada no diretório {spectrogram_dir}. Verifique os arquivos de áudio fornecidos.")
        return

    # Treinar o modelo com os espectrogramas gerados
    model, classes = train_model(
        spectrogram_dir, num_classes, model_name, epochs, learning_rate, batch_size, train_split, valid_split
    )

    # Salvar o modelo treinado
    model_save_path = "trained_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Modelo salvo em {model_save_path}")

    # Salvar as classes em um arquivo
    classes_save_path = "classes.txt"
    with open(classes_save_path, "w") as f:
        for class_name in classes:
            f.write(f"{class_name}\n")
    print(f"Classes salvas em {classes_save_path}")

if __name__ == "__main__":
    main()

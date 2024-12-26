import os
import glob

import streamlit as st
import torch
import torchaudio
import torchaudio.transforms as T

from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import random

# ------------------------------------------------------------------------------
# Dataset customizado para ler áudios .wav organizados em subpastas (classes)
# ------------------------------------------------------------------------------
class AudioFolderDataset(Dataset):
    """
    Lê arquivos de áudio (formato .wav) de subpastas, cada subpasta representando uma classe.
    Exemplo de estrutura:
      data/
        classeA/
          audioA1.wav
          audioA2.wav
        classeB/
          audioB1.wav
    """
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        # Obter classes a partir das subpastas
        self.classes = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Lista de (caminho_arquivo, idx_classe)
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            # Pegar todos .wav daquela classe
            files = glob.glob(os.path.join(cls_dir, '*.wav'))
            for f in files:
                self.samples.append((f, self.class_to_idx[cls_name]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Nenhum arquivo .wav encontrado em {root_dir}. "
                f"Verifique se há subpastas com áudios .wav."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        audio_path, label = self.samples[index]

        # Carregar áudio (waveform, sample_rate)
        waveform, sr = torchaudio.load(audio_path)

        # Se tiver transformações definidas (ex: MelSpectrogram)
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label


# ------------------------------------------------------------------------------
# Exemplo de um modelo simples (CNN) para classificação de espectrogramas
# ------------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """
    Se você converter o waveform para espectrograma (shape = [N_MELS, tempo]),
    pode tratar como uma "imagem" de 1 canal e treinar uma CNN simples.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # Exemplo de 1 conv + pooling + FC
        # Ajuste as camadas conforme o tamanho do espectrograma.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # Saída do conv ~ depende do tamanho do espectrograma
        self.fc = nn.Sequential(
            nn.Linear(16 * 16 * 16, 64),  # Exemplo: 16 filtros de 16x16
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x deve ter shape (B, 1, freq, tempo)
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ------------------------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_loop(dataloader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        wave, labels = batch
        # wave shape -> (B, freq, tempo) se transform for MelSpectrogram
        # Precisamos de shape (B, 1, freq, tempo) p/ CNN
        wave = wave.unsqueeze(1).to(device)  # (B,1,freq,tempo)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(wave)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_loop(dataloader, model, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            wave, labels = batch
            wave = wave.unsqueeze(1).to(device)
            labels = labels.to(device)

            outputs = model(wave)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, acc


# ------------------------------------------------------------------------------
# Função principal de treino
# ------------------------------------------------------------------------------
def train_model(root_dir,
                num_classes=2,
                epochs=10,
                learning_rate=1e-3,
                batch_size=16,
                train_split=0.8,
                device='cpu'):
    # Definir transform de mel-spectrogram (exemplo)
    # Esse "torchaudio.transforms.MelSpectrogram" converte wave -> espectrograma
    # Ajuste conforme sr.
    mel_transform = T.MelSpectrogram(
        sample_rate=16000,  # Ajuste ao sample rate real
        n_fft=1024,
        hop_length=256,
        n_mels=64
    )

    # Dataset
    full_dataset = AudioFolderDataset(root_dir=root_dir, transform=mel_transform)
    if len(full_dataset.classes) < num_classes:
        raise ValueError(f"As classes detectadas ({full_dataset.classes}) "
                         f"são menos que num_classes={num_classes} configurado.")

    # Split train/val
    total_len = len(full_dataset)
    train_len = int(total_len * train_split)
    val_len = total_len - train_len
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len])

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Modelo
    model = SimpleCNN(num_classes=num_classes).to(device)

    # Otimizador e loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss = train_loop(train_loader, model, criterion, optimizer, device)
        val_loss, val_acc = eval_loop(val_loader, model, criterion, device)
        print(f"[{epoch+1}/{epochs}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.2f}")

    return model


# ------------------------------------------------------------------------------
# Exemplo de uso com Streamlit
# ------------------------------------------------------------------------------
def main():
    st.title("Classificador de Áudio (Exemplo)")

    # Parâmetros
    root_dir = st.text_input("Caminho raiz (pastas de áudio):", "data")
    epochs = st.number_input("Épocas", 1, 100, 10)
    batch_size = st.number_input("Batch size", 1, 128, 16)
    lr = st.number_input("Learning Rate", 1e-6, 1.0, 1e-3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if st.button("Treinar"):
        try:
            model = train_model(
                root_dir=root_dir,
                num_classes=2,  # Ajuste para o nº de classes que você tiver
                epochs=epochs,
                learning_rate=lr,
                batch_size=batch_size,
                train_split=0.8,
                device=device
            )
            st.success("Treino concluído com sucesso!")
            st.write("Modelo treinado:", model)
        except Exception as e:
            st.error(f"Erro ao treinar: {e}")

if __name__ == "__main__":
    set_seed(42)
    main()

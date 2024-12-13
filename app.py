# app.py
import streamlit as st
import os
import zipfile
import tempfile
import shutil
import numpy as np
import pandas as pd
import librosa
import librosa.display
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import gc

# Configurações para visualizações
sns.set(style='whitegrid', context='notebook')

# ==================== CONTROLE DE REPRODUTIBILIDADE ====================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random_state = SEED
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ==================== DEFINIÇÃO DAS TRANSFORMAÇÕES DE DATA AUGMENTATION ====================
def get_shift_transform():
    """
    Retorna a transformação Shift adequada conforme a versão do audiomentations.
    """
    try:
        # Tenta usar min_fraction e max_fraction
        return Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5)
    except TypeError:
        # Caso contrário, usa min_shift e max_shift
        return Shift(min_shift=-0.5, max_shift=0.5, p=0.5)

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    get_shift_transform(),
])

# ==================== DEFINIÇÃO DAS FUNÇÕES DE PROCESSAMENTO ====================
def load_audio(file_path, sr=None):
    """
    Carrega um arquivo de áudio.
    
    Parameters:
    - file_path (str): Caminho para o arquivo de áudio.
    - sr (int, optional): Taxa de amostragem. Se None, usa a taxa original.
    
    Returns:
    - data (np.ndarray): Sinal de áudio.
    - sr (int): Taxa de amostragem.
    """
    try:
        data, sr = librosa.load(file_path, sr=sr, res_type='kaiser_fast')
        return data, sr
    except Exception as e:
        st.error(f"Erro ao carregar o áudio {file_path}: {e}")
        return None, None

def extract_features(data, sr):
    """
    Extrai os MFCCs do sinal de áudio e calcula a média ao longo do tempo.
    
    Parameters:
    - data (np.ndarray): Sinal de áudio.
    - sr (int): Taxa de amostragem.
    
    Returns:
    - mfccs_scaled (np.ndarray): Vetor de características MFCC.
    """
    try:
        mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        st.error(f"Erro ao extrair MFCC: {e}")
        return None

def augment_audio(data, sr):
    """
    Aplica Data Augmentation ao sinal de áudio.
    
    Parameters:
    - data (np.ndarray): Sinal de áudio.
    - sr (int): Taxa de amostragem.
    
    Returns:
    - augmented_data (np.ndarray): Sinal de áudio aumentado.
    """
    try:
        augmented_data = augment(samples=data, sample_rate=sr)
        return augmented_data
    except Exception as e:
        st.error(f"Erro ao aplicar Data Augmentation: {e}")
        return data  # Retorna o original em caso de erro

# ==================== DEFINIÇÃO DO DATASET PERSONALIZADO ====================
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        """
        Dataset personalizado para carregar arquivos de áudio.
        
        Args:
            file_paths (list): Lista de caminhos para os arquivos de áudio.
            labels (list): Lista de rótulos correspondentes.
            transform (callable, optional): Transformação a ser aplicada no áudio.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        data, sr = load_audio(file_path, sr=None)
        if data is None:
            # Retorna zeros se o áudio não puder ser carregado
            mfccs = np.zeros(40)
        else:
            mfccs = extract_features(data, sr)
            if mfccs is None:
                mfccs = np.zeros(40)
            if self.transform:
                augmented_data = augment_audio(data, sr)
                mfccs = extract_features(augmented_data, sr)
                if mfccs is None:
                    mfccs = np.zeros(40)
        mfccs = torch.tensor(mfccs, dtype=torch.float32)
        return mfccs, label

# ==================== DEFINIÇÃO DO MODELO CNN ====================
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=10, activation='relu')
        self.dropout1 = nn.Dropout(0.4)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10, padding=5)
        self.dropout2 = nn.Dropout(0.4)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 10, 64)  # Ajuste este valor conforme o tamanho dos MFCCs
        self.dropout3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# ==================== FUNÇÃO PRINCIPAL DA APLICAÇÃO STREAMLIT ====================
def main():
    # Definir a página do Streamlit
    st.set_page_config(page_title="Classificação de Sons", layout="wide", page_icon="🔊")
    
    # Layout da página
    st.title("Classificação de Sons de Água Vibrando em Copo de Vidro com Data Augmentation e CNN 🔊")
    st.write("""
    Este aplicativo permite treinar um modelo de classificação de sons de diferentes tipos de água vibrando em copos de vidro.
    Você pode:
    - Carregar um dataset de áudio organizado em pastas por classe.
    - Realizar Data Augmentation para aumentar a diversidade dos dados.
    - Treinar uma Rede Neural Convolucional (CNN).
    - Avaliar o modelo com métricas detalhadas.
    - Classificar novas amostras de áudio.
    """)
    
    # Barra Lateral de Configurações
    st.sidebar.title("Configurações do Treinamento")
    
    # Upload do arquivo ZIP com os dados
    uploaded_zip = st.sidebar.file_uploader("Upload do Dataset (ZIP)", type=["zip"], key="zip_uploader")
    
    # Parâmetros do modelo
    num_classes = st.sidebar.number_input("Número de Classes:", min_value=2, step=1, key="num_classes")
    epochs = st.sidebar.slider("Número de Épocas:", min_value=10, max_value=500, value=200, step=10, key="epochs")
    batch_size = st.sidebar.selectbox("Tamanho do Lote:", options=[16, 32, 64, 128], index=1, key="batch_size")
    learning_rate = st.sidebar.select_slider("Taxa de Aprendizagem:", options=[0.1, 0.01, 0.001, 0.0001], value=0.001, key="learning_rate")
    l2_lambda = st.sidebar.number_input("Regularização L2 (Weight Decay):", min_value=0.0, max_value=0.1, value=0.01, step=0.01, key="l2_lambda")
    patience = st.sidebar.number_input("Paciência para Early Stopping:", min_value=1, max_value=20, value=5, step=1, key="patience")
    
    # Botão para iniciar o treinamento
    if st.sidebar.button("Iniciar Treinamento", key="train_button"):
        if uploaded_zip is not None and num_classes >= 2:
            with st.spinner("Processando o dataset..."):
                # Cria um diretório temporário
                temp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(temp_dir, "dataset.zip")
                
                # Salva o arquivo ZIP
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.read())
                
                # Extrai o ZIP
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Define base_path como temp_dir, assumindo que as classes estão diretamente nele
                base_path = temp_dir
                categories = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
                
                if len(categories) < num_classes:
                    st.error(f"O número de classes encontradas ({len(categories)}) é menor que o número especificado ({num_classes}).")
                    shutil.rmtree(temp_dir)
                    return
                
                st.success(f"Classes encontradas: {categories}")
                
                # Coleta os caminhos dos arquivos e labels
                file_paths = []
                labels = []
                for cat in categories:
                    cat_path = os.path.join(base_path, cat)
                    files_in_cat = [f for f in os.listdir(cat_path) if f.lower().endswith('.wav')]
                    if len(files_in_cat) == 0:
                        st.warning(f"A classe '{cat}' não possui arquivos .wav.")
                    for file_name in files_in_cat:
                        full_path = os.path.join(cat_path, file_name)
                        file_paths.append(full_path)
                        labels.append(cat)
                
                df = pd.DataFrame({'file_path': file_paths, 'class': labels})
                st.write("**Dataset:**")
                st.dataframe(df.head())
                st.write(f"Total de amostras: {len(df)}")
                
                # Codificação das classes
                labelencoder = LabelEncoder()
                y = labelencoder.fit_transform(df['class'])
                
                # Extração de features
                st.write("**Extraindo features (MFCCs)...**")
                X = []
                y_valid = []
                for i, row in df.iterrows():
                    file = row['file_path']
                    data, sr = load_audio(file, sr=None)
                    if data is not None:
                        features = extract_features(data, sr)
                        if features is not None:
                            X.append(features)
                            y_valid.append(y[i])
                        else:
                            st.warning(f"Erro na extração de features do arquivo: {file}")
                    else:
                        st.warning(f"Erro no carregamento do arquivo: {file}")
                
                X = np.array(X)
                y_valid = np.array(y_valid)
                st.write(f"Features extraídas: {X.shape}")
                
                # Divisão dos dados
                st.write("**Dividindo os dados em Treino e Teste...**")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_valid, test_size=0.2, random_state=random_state, stratify=y_valid)
                st.write(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
                
                # Data Augmentation
                st.write("**Aplicando Data Augmentation...**")
                augment_factor = 10  # Fator de aumento
                X_train_augmented = []
                y_train_augmented = []
                for i in range(len(X_train)):
                    file = df['file_path'].iloc[i]
                    data, sr = load_audio(file, sr=None)
                    if data is not None:
                        for _ in range(augment_factor):
                            augmented_data = augment_audio(data, sr)
                            if augmented_data is not None:
                                features = extract_features(augmented_data, sr)
                                if features is not None:
                                    X_train_augmented.append(features)
                                    y_train_augmented.append(y_train[i])
                X_train_augmented = np.array(X_train_augmented)
                y_train_augmented = np.array(y_train_augmented)
                st.write(f"Dados aumentados: {X_train_augmented.shape}")
                
                # Combinação dos dados originais e aumentados
                X_train_combined = np.concatenate((X_train, X_train_augmented), axis=0)
                y_train_combined = np.concatenate((y_train, y_train_augmented), axis=0)
                st.write(f"Treino combinado: {X_train_combined.shape}")
                
                # Divisão em treino final e validação
                st.write("**Dividindo o treino combinado em Treino Final e Validação...**")
                X_train_final, X_val, y_train_final, y_val = train_test_split(
                    X_train_combined, y_train_combined, test_size=0.1, random_state=random_state, stratify=y_train_combined)
                st.write(f"Treino Final: {X_train_final.shape}, Validação: {X_val.shape}")
                
                # Ajuste das formas para o modelo CNN (Conv1D)
                X_train_final = X_train_final.reshape((X_train_final.shape[0], X_train_final.shape[1], 1))
                X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                
                # Cálculo de class weights
                st.write("**Calculando class weights...**")
                class_weights = compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(y_train_final),
                    y=y_train_final
                )
                class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
                st.write(f"Class weights: {class_weight_dict}")
                
                # Criação dos datasets e dataloaders
                st.write("**Criando datasets e dataloaders...**")
                train_dataset = AudioDataset(file_paths=X_train_final, labels=y_train_final, transform=None)
                val_dataset = AudioDataset(file_paths=X_val, labels=y_val, transform=None)
                test_dataset = AudioDataset(file_paths=X_test, labels=y_test, transform=None)
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                
                # Inicialização do modelo
                st.write("**Inicializando o modelo CNN...**")
                model = CNNModel(num_classes=num_classes)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                
                # Definição da função de perda e otimizador
                criterion = nn.CrossEntropyLoss(weight=torch.tensor(list(class_weight_dict.values())).to(device))
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
                
                # Callbacks simulados (já que Streamlit não suporta callbacks diretamente)
                best_val_loss = float('inf')
                epochs_no_improve = 0
                best_model_wts = None
                history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
                
                # Treinamento do modelo
                st.write("**Iniciando o treinamento do modelo...**")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for epoch in range(epochs):
                    model.train()
                    running_loss = 0.0
                    running_corrects = 0
                    total = 0
                    
                    for inputs, labels in train_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        total += inputs.size(0)
                    
                    epoch_loss = running_loss / total
                    epoch_acc = running_corrects.double() / total
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc.item())
                    
                    # Validação
                    model.eval()
                    val_running_loss = 0.0
                    val_running_corrects = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for inputs, labels in val_loader:
                            inputs = inputs.to(device)
                            labels = labels.to(device)
                            
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                            
                            val_running_loss += loss.item() * inputs.size(0)
                            val_running_corrects += torch.sum(preds == labels.data)
                            val_total += inputs.size(0)
                    
                    val_loss = val_running_loss / val_total
                    val_acc = val_running_corrects.double() / val_total
                    history['val_loss'].append(val_loss)
                    history['val_acc'].append(val_acc.item())
                    
                    # Atualização da barra de progresso e status
                    progress_bar.progress((epoch + 1) / epochs)
                    status_text.text(f'Época {epoch+1}/{epochs} - Treino Loss: {epoch_loss:.4f} | Treino Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
                    
                    # Early Stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        best_model_wts = model.state_dict()
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= patience:
                            st.write("Early Stopping: Não houve melhora na perda de validação.")
                            break
                
                # Carregar os melhores pesos
                if best_model_wts is not None:
                    model.load_state_dict(best_model_wts)
                    st.success("Melhores pesos do modelo carregados.")
                
                # Avaliação no conjunto de teste
                st.write("**Avaliando o modelo no conjunto de teste...**")
                model.eval()
                test_running_corrects = 0
                test_total = 0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        
                        test_running_corrects += torch.sum(preds == labels.data)
                        test_total += inputs.size(0)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                test_acc = test_running_corrects.double() / test_total
                st.write(f"Acurácia no Teste: {test_acc.item() * 100:.2f}%")
                
                # Matriz de Confusão
                st.write("**Matriz de Confusão:**")
                cm = confusion_matrix(all_labels, all_preds)
                cm_df = pd.DataFrame(cm, index=labelencoder.classes_, columns=labelencoder.classes_)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Classe Prevista')
                ax.set_ylabel('Classe Real')
                st.pyplot(fig)
                plt.close(fig)
                
                # Relatório de Classificação
                st.write("**Relatório de Classificação:**")
                report = classification_report(all_labels, all_preds, target_names=labelencoder.classes_, zero_division=0)
                st.text(report)
                
                # Salvamento do Modelo
                st.write("**Salvando o modelo treinado...**")
                model_path = os.path.join(temp_dir, "modelo_trained.pth")
                torch.save(model.state_dict(), model_path)
                st.download_button(
                    label="Download do Modelo Treinado",
                    data=open(model_path, "rb").read(),
                    file_name="modelo_trained.pth",
                    mime="application/octet-stream",
                )
                
                # Salvamento das Classes
                classes_path = os.path.join(temp_dir, "classes.txt")
                with open(classes_path, "w") as f:
                    for cls in labelencoder.classes_:
                        f.write(f"{cls}\n")
                st.download_button(
                    label="Download das Classes",
                    data=open(classes_path, "rb").read(),
                    file_name="classes.txt",
                    mime="text/plain",
                )
                
                # Limpeza
                shutil.rmtree(temp_dir)
                gc.collect()
                st.success("Treinamento concluído e arquivos temporários removidos.")
        else:
            st.error("Por favor, faça o upload do dataset e especifique pelo menos 2 classes.")
    
    # ==================== CLASSIFICAÇÃO DE NOVOS ÁUDIO ====================
    st.header("Classificação de Novas Amostras de Áudio")
    
    # Upload de um novo arquivo de áudio para classificação
    uploaded_audio = st.file_uploader("Faça upload de um arquivo de áudio (.wav) para classificação", type=["wav"], key="audio_uploader")
    
    if uploaded_audio is not None:
        # Salva o arquivo de áudio temporariamente
        temp_audio_path = tempfile.mktemp(suffix=".wav")
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_audio.read())
        
        # Exibe o áudio
        st.audio(uploaded_audio, format='audio/wav')
        
        # Carrega o áudio e extrai features
        data, sr = load_audio(temp_audio_path, sr=None)
        if data is not None:
            mfccs = extract_features(data, sr)
            if mfccs is not None:
                # Reshape para compatibilidade com o modelo
                mfccs = mfccs.reshape(1, -1, 1)
                mfccs_tensor = torch.tensor(mfccs, dtype=torch.float32).to(device)
                
                # Carrega o modelo treinado
                with st.spinner("Classificando o áudio..."):
                    # Verifica se o modelo foi treinado
                    if 'model' in locals():
                        model.eval()
                        with torch.no_grad():
                            outputs = model(mfccs_tensor)
                            _, preds = torch.max(outputs, 1)
                            pred_label = labelencoder.inverse_transform(preds.cpu().numpy())[0]
                            confidence = torch.max(outputs, 1)[0].cpu().numpy()[0]
                        
                        st.success(f"Classe Predita: {pred_label} com confiança de {confidence * 100:.2f}%")
                        
                        # Exibe as probabilidades das classes
                        class_probs = outputs.cpu().numpy()[0]
                        class_probs_dict = {labelencoder.classes_[i]: float(class_probs[i]) for i in range(len(labelencoder.classes_))}
                        st.write("**Probabilidades das Classes:**")
                        st.write(class_probs_dict)
                        
                        # Visualizações
                        st.write("**Visualizações do Áudio:**")
                        fig, axs = plt.subplots(4, 1, figsize=(14, 20))
                        
                        # Forma de Onda
                        librosa.display.waveshow(data, sr=sr, ax=axs[0])
                        axs[0].set_title("Forma de Onda")
                        axs[0].set_xlabel("Tempo (s)")
                        axs[0].set_ylabel("Amplitude")
                        
                        # Espectro de Frequências
                        D = np.abs(librosa.stft(data))
                        DB = librosa.amplitude_to_db(D, ref=np.max)
                        img = librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='hz', ax=axs[1], cmap='magma')
                        axs[1].set_title("Espectro de Frequências")
                        fig.colorbar(img, ax=axs[1], format="%+2.0f dB")
                        
                        # Spectrograma STFT
                        librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='hz', ax=axs[2], cmap='magma')
                        axs[2].set_title("Spectrograma STFT")
                        fig.colorbar(img, ax=axs[2], format="%+2.0f dB")
                        
                        # Spectrograma MFCC
                        mfccs_plot = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
                        mfccs_db_plot = librosa.amplitude_to_db(np.abs(mfccs_plot))
                        img_mfcc = librosa.display.specshow(mfccs_db_plot, x_axis='time', y_axis='mel', sr=sr, ax=axs[3], cmap='Spectral')
                        axs[3].set_title("Spectrograma MFCC")
                        fig.colorbar(img_mfcc, ax=axs[3], format="%+2.f dB")
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.error("Modelo não encontrado. Por favor, treine o modelo antes de realizar classificações.")
            else:
                st.error("Erro na extração de features do áudio.")
        else:
            st.error("Erro no carregamento do áudio.")
        
        # Remoção do arquivo temporário
        os.remove(temp_audio_path)

# ==================== EXECUÇÃO DO SCRIPT ====================
if __name__ == "__main__":
    main()

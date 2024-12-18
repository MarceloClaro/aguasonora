import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, KFold
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import streamlit as st
import tempfile
from PIL import Image, UnidentifiedImageError
import io
import torch
import zipfile
import gc
import os
import logging
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import shap
import torchvision.transforms as torch_transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset

# Configuração de logging
logging.basicConfig(
    filename='experiment_logs.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Configurações de semente para reprodutibilidade
def set_seeds(seed):
    """
    Define as sementes para reprodutibilidade.

    Args:
        seed (int): Valor da semente.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Funções auxiliares
def carregar_audio(caminho_arquivo, sr=None):
    """
    Carrega um arquivo de áudio usando Librosa.

    Args:
        caminho_arquivo (str): Caminho para o arquivo de áudio.
        sr (int, optional): Taxa de amostragem. Se None, usa a taxa original.

    Returns:
        tuple: Dados de áudio e taxa de amostragem.
    """
    try:
        data, sr = librosa.load(caminho_arquivo, sr=sr, res_type='kaiser_fast', backend='soundfile')
        return data, sr
    except Exception as e:
        logging.error(f"Erro ao carregar áudio {caminho_arquivo}: {e}")
        return None, None

def extrair_features(data, sr, use_mfcc=True, use_spectral_centroid=True):
    """
    Extrai features do áudio.

    Args:
        data (np.array): Dados de áudio.
        sr (int): Taxa de amostragem.
        use_mfcc (bool): Se True, extrai MFCCs.
        use_spectral_centroid (bool): Se True, extrai centróide espectral.

    Returns:
        np.array: Vetor de features normalizado.
    """
    try:
        features_list = []
        if use_mfcc:
            mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            features_list.append(mfccs_scaled)
        if use_spectral_centroid:
            centroid = librosa.feature.spectral_centroid(y=data, sr=sr)
            centroid_mean = np.mean(centroid, axis=1)
            features_list.append(centroid_mean)
        if len(features_list) > 1:
            features_vector = np.concatenate(features_list, axis=0)
        else:
            features_vector = features_list[0]
        # Normalização
        features_vector = (features_vector - np.mean(features_vector)) / (np.std(features_vector) + 1e-9)
        return features_vector
    except Exception as e:
        logging.error(f"Erro ao extrair features: {e}")
        return None

def aumentar_audio(data, sr, augmentations):
    """
    Aplica aumentação de dados no áudio.

    Args:
        data (np.array): Dados de áudio.
        sr (int): Taxa de amostragem.
        augmentations (Compose): Pipeline de aumentação.

    Returns:
        np.array: Áudio aumentado.
    """
    try:
        return augmentations(samples=data, sample_rate=sr)
    except Exception as e:
        logging.error(f"Erro ao aumentar áudio: {e}")
        return data

def gerar_espectrograma(data, sr):
    """
    Gera um espectrograma a partir do áudio.

    Args:
        data (np.array): Dados de áudio.
        sr (int): Taxa de amostragem.

    Returns:
        PIL.Image: Imagem do espectrograma.
    """
    try:
        S = librosa.stft(data, n_fft=1024, hop_length=512)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        plt.figure(figsize=(2,2), dpi=100)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='gray')
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img
    except Exception as e:
        logging.error(f"Erro ao gerar espectrograma: {e}")
        return None

def visualizar_audio(data, sr):
    """
    Visualiza diferentes representações do áudio.

    Args:
        data (np.array): Dados de áudio.
        sr (int): Taxa de amostragem.
    """
    try:
        # Forma de onda no tempo
        fig_wave, ax_wave = plt.subplots(figsize=(8,4))
        librosa.display.waveshow(data, sr=sr, ax=ax_wave)
        ax_wave.set_title("Forma de Onda no Tempo")
        ax_wave.set_xlabel("Tempo (s)")
        ax_wave.set_ylabel("Amplitude")
        st.pyplot(fig_wave)
        plt.close(fig_wave)

        # Espectro de frequência
        fft = np.fft.fft(data)
        fft_abs = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(data), 1/sr)[:len(fft)//2]
        fig_fft, ax_fft = plt.subplots(figsize=(8,4))
        ax_fft.plot(freqs, fft_abs)
        ax_fft.set_title("Espectro (Amplitude x Frequência)")
        ax_fft.set_xlabel("Frequência (Hz)")
        ax_fft.set_ylabel("Amplitude")
        st.pyplot(fig_fft)
        plt.close(fig_fft)

        # Espectrograma
        D = np.abs(librosa.stft(data))**2
        S = librosa.power_to_db(D, ref=np.max)
        fig_spec, ax_spec = plt.subplots(figsize=(8,4))
        img_spec = librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='hz', ax=ax_spec)
        ax_spec.set_title("Espectrograma")
        fig_spec.colorbar(img_spec, ax=ax_spec, format='%+2.0f dB')
        st.pyplot(fig_spec)
        plt.close(fig_spec)

        # MFCCs
        mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
        fig_mfcc, ax_mfcc = plt.subplots(figsize=(8,4))
        img_mfcc = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax_mfcc)
        ax_mfcc.set_title("MFCCs")
        fig_mfcc.colorbar(img_mfcc, ax=ax_mfcc)
        st.pyplot(fig_mfcc)
        plt.close(fig_mfcc)
    except Exception as e:
        logging.error(f"Erro ao visualizar áudio: {e}")

def visualizar_exemplos_classe(df, y, classes, augmentation=False, sr=22050, metodo='CNN'):
    """
    Visualiza exemplos de cada classe, tanto originais quanto aumentados.

    Args:
        df (pd.DataFrame): DataFrame com caminhos dos arquivos e classes.
        y (np.array): Labels codificados.
        classes (list): Lista de classes.
        augmentation (bool, optional): Se True, mostra exemplos aumentados.
        sr (int, optional): Taxa de amostragem.
        metodo (str, optional): Método de treinamento utilizado.
    """
    classes_indices = {c: np.where(y == i)[0] for i, c in enumerate(classes)}
    st.markdown("### Visualizações Espectrais e MFCCs de Exemplos do Dataset (1 de cada classe original e 1 de cada classe aumentada)")
    
    transforms_aug = None
    if metodo in ['CNN Personalizada', 'ResNet18', 'ResNet-18']:
        transforms_aug = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
        ])
    
    for c in classes:
        st.markdown(f"#### Classe: {c}")
        indices_classe = classes_indices[c]
        if len(indices_classe) == 0:
            logging.warning(f"Nenhuma amostra encontrada para a classe: {c}")
            continue
        idx_original = random.choice(indices_classe)
        arquivo_original = df.iloc[idx_original]['caminho_arquivo']
        data_original, sr_original = carregar_audio(arquivo_original, sr=None)
        if data_original is not None and sr_original is not None:
            visualizar_audio(data_original, sr_original)
            if metodo in ['ResNet18', 'ResNet-18']:
                espectrograma = gerar_espectrograma(data_original, sr_original)
                if espectrograma:
                    st.image(espectrograma, caption="Espectrograma Original", use_column_width=True)
        else:
            logging.warning(f"Dados inválidos para a amostra original da classe: {c}")
        
        if augmentation and transforms_aug is not None:
            try:
                if len(indices_classe) > 1:
                    idx_aug = random.choice(indices_classe)
                else:
                    idx_aug = idx_original
                arquivo_aug = df.iloc[idx_aug]['caminho_arquivo']
                data_aug, sr_aug = carregar_audio(arquivo_aug, sr=None)
                if data_aug is not None and sr_aug is not None:
                    aug_data = aumentar_audio(data_aug, sr_aug, transforms_aug)
                    visualizar_audio(aug_data, sr_aug)
                    if metodo in ['ResNet18', 'ResNet-18']:
                        espectrograma_aug = gerar_espectrograma(aug_data, sr_aug)
                        if espectrograma_aug:
                            st.image(espectrograma_aug, caption="Espectrograma Aumentado", use_column_width=True)
            except Exception as e:
                logging.error(f"Erro ao aumentar áudio para a classe {c}: {e}")

def escolher_k_kmeans(X_original, y, max_k=10):
    """
    Determina o melhor número de clusters usando o coeficiente de silhueta.

    Args:
        X_original (np.array): Features.
        y (np.array): Labels.
        max_k (int, optional): Número máximo de clusters a considerar.

    Returns:
        int: Melhor valor de k.
    """
    melhor_k = 2
    melhor_sil = -1
    n_amostras = X_original.shape[0]
    max_k = min(max_k, n_amostras-1)
    if max_k < 2:
        max_k = 2
    for k in range(2, max_k+1):
        kmeans_test = KMeans(n_clusters=k, random_state=42)
        labels_test = kmeans_test.fit_predict(X_original)
        if len(np.unique(labels_test)) > 1:
            sil = silhouette_score(X_original, labels_test)
            if sil > melhor_sil:
                melhor_sil = sil
                melhor_k = k
    return melhor_k

class AudioSpectrogramDataset(Dataset):
    """
    Dataset personalizado para espectrogramas de áudio.

    Args:
        df (pd.DataFrame): DataFrame com caminhos dos arquivos e classes.
        classes (list): Lista de classes.
        transform (callable, optional): Transformação a ser aplicada nas imagens.
    """
    def __init__(self, df, classes, transform=None):
        self.df = df
        self.classes = classes
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(classes)
        self.indices_validos = self._filtrar_amostras_validas()
    
    def _filtrar_amostras_validas(self):
        """Filtra amostras válidas antes de criar o dataset."""
        indices_validos = []
        for idx in range(len(self.df)):
            arquivo = self.df.iloc[idx]['caminho_arquivo']
            classe = self.df.iloc[idx]['classe']
            data, sr = carregar_audio(arquivo, sr=None)
            if data is None or sr is None:
                logging.warning(f"Dados inválidos no índice {idx}: {arquivo}")
                continue
            espectrograma = gerar_espectrograma(data, sr)
            if espectrograma is None:
                logging.warning(f"Espectrograma inválido no índice {idx}: {arquivo}")
                continue
            indices_validos.append(idx)
        return indices_validos
    
    def __len__(self):
        return len(self.indices_validos)
    
    def __getitem__(self, idx):
        real_idx = self.indices_validos[idx]
        try:
            arquivo = self.df.iloc[real_idx]['caminho_arquivo']
            classe = self.df.iloc[real_idx]['classe']
            data, sr = carregar_audio(arquivo, sr=None)
            if data is None or sr is None:
                logging.error(f"Não foi possível carregar o áudio no índice {real_idx}: {arquivo}")
                raise ValueError(f"Dados inválidos no índice {real_idx}: {arquivo}")
            espectrograma = gerar_espectrograma(data, sr)
            if espectrograma is None:
                logging.error(f"Não foi possível gerar espectrograma no índice {real_idx}: {arquivo}")
                raise ValueError(f"Espectrograma inválido no índice {real_idx}: {arquivo}")
            if self.transform:
                espectrograma = self.transform(espectrograma)
                if espectrograma is None:
                    logging.error(f"Transformação retornou None para o índice {real_idx}: {arquivo}")
                    raise ValueError(f"Transformação inválida no índice {real_idx}: {arquivo}")
            label = self.label_encoder.transform([classe])[0]
            return espectrograma, label
        except Exception as e:
            logging.error(f"Erro em __getitem__ para o índice {real_idx}: {e}")
            # Em vez de retornar None, levanta a exceção para que possa ser capturada no treinamento
            raise e

def custom_collate(batch):
    """
    Função de colagem personalizada para ignorar amostras inválidas.

    Args:
        batch (list): Lista de amostras.

    Returns:
        batch: Batch processado.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def verificar_contagem_classes(y, k_folds=1):
    """
    Verifica se todas as classes têm pelo menos k_folds amostras.

    Args:
        y (np.array): Labels.
        k_folds (int, optional): Número de folds.

    Returns:
        bool: True se todas as classes tiverem amostras suficientes, False caso contrário.
    """
    classes, counts = np.unique(y, return_counts=True)
    insufficient_classes = classes[counts < k_folds]
    if len(insufficient_classes) > 0:
        logging.warning(f"Classes com menos de {k_folds} amostras: {insufficient_classes}")
        return False
    return True

# Funções principais
def classificar_audio(SEED):
    """
    Função para classificar novos áudios usando um modelo treinado.

    Args:
        SEED (int): Valor da semente para reprodutibilidade.
    """
    with st.expander("Classificação de Novo Áudio com Modelo Treinado"):
        metodo_classificacao = st.selectbox(
            "Escolha o Método de Classificação:",
            ["CNN Personalizada", "ResNet-18"],
            key='select_classification_method'
        )

        modelo_file = st.file_uploader("Upload do Modelo (.keras, .h5, .pth)", type=["keras","h5","pth"], key='model_uploader')
        classes_file = st.file_uploader("Upload do Arquivo de Classes (classes.txt)", type=["txt"], key='classes_uploader')

        if modelo_file is not None and classes_file is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(modelo_file.name)[1]) as tmp_model:
                    tmp_model.write(modelo_file.read())
                    caminho_modelo = tmp_model.name

                if metodo_classificacao == "CNN Personalizada":
                    modelo = tf.keras.models.load_model(caminho_modelo, compile=False)
                elif metodo_classificacao == "ResNet-18":
                    modelo = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                    modelo.fc = torch.nn.Linear(modelo.fc.in_features, len(classes))  # Atualizado para o número correto de classes
                logging.info("Modelo carregado com sucesso.")
                st.success("Modelo carregado com sucesso!")
            except Exception as e:
                st.error(f"Erro ao carregar o modelo: {e}")
                logging.error(f"Erro ao carregar o modelo: {e}")
                return

            try:
                classes = classes_file.read().decode("utf-8").splitlines()
                classes = [c.strip() for c in classes if c.strip()]
                if not classes:
                    st.error("O arquivo de classes está vazio.")
                    return
                logging.info("Arquivo de classes carregado com sucesso.")
            except Exception as e:
                st.error(f"Erro ao ler o arquivo de classes: {e}")
                logging.error(f"Erro ao ler o arquivo de classes: {e}")
                return

            if metodo_classificacao == "CNN Personalizada":
                try:
                    num_classes_model = modelo.output_shape[-1]
                    num_classes_file = len(classes)
                    if num_classes_file != num_classes_model:
                        st.error(f"Número de classes ({num_classes_file}) não corresponde ao número de saídas do modelo ({num_classes_model}).")
                        logging.error(f"Número de classes ({num_classes_file}) não corresponde ao número de saídas do modelo ({num_classes_model}).")
                        return
                except AttributeError as e:
                    st.error(f"Erro ao verificar a saída do modelo: {e}")
                    logging.error(f"Erro ao verificar a saída do modelo: {e}")
                    return
            elif metodo_classificacao == "ResNet-18":
                try:
                    modelo.fc = torch.nn.Linear(modelo.fc.in_features, len(classes))
                    modelo = modelo.to('cuda' if torch.cuda.is_available() else 'cpu')
                    logging.info("Última camada da ResNet-18 ajustada para o número de classes.")
                    st.success("Última camada da ResNet-18 ajustada para o número de classes.")
                except Exception as e:
                    st.error(f"Erro ao ajustar a ResNet-18: {e}")
                    logging.error(f"Erro ao ajustar a ResNet-18: {e}")
                    return

            st.markdown("**Modelo e Classes Carregados!**")
            st.markdown(f"**Classes:** {', '.join(classes)}")

            audio_file = st.file_uploader("Upload do Áudio para Classificação", type=["wav","mp3","flac","ogg","m4a"], key='audio_uploader')
            if audio_file is not None:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_audio:
                        tmp_audio.write(audio_file.read())
                        caminho_audio = tmp_audio.name

                    data, sr = carregar_audio(caminho_audio, sr=None)
                    if data is not None:
                        if metodo_classificacao == "CNN Personalizada":
                            ftrs = extrair_features(data, sr, use_mfcc=True, use_spectral_centroid=True)
                            if ftrs is not None:
                                ftrs = ftrs.reshape(1, -1, 1)
                                pred = modelo.predict(ftrs)
                                pred_class = np.argmax(pred, axis=1)
                                pred_label = classes[pred_class[0]]
                                confidence = pred[0][pred_class[0]] * 100
                                st.markdown(f"**Classe Predita:** {pred_label} (Confiança: {confidence:.2f}%)")

                                fig_prob, ax_prob = plt.subplots(figsize=(8,4))
                                ax_prob.bar(classes, pred[0], color='skyblue')
                                ax_prob.set_title("Probabilidades por Classe")
                                ax_prob.set_ylabel("Probabilidade")
                                plt.xticks(rotation=45)
                                st.pyplot(fig_prob)
                                plt.close(fig_prob)

                                st.audio(caminho_audio)
                                visualizar_audio(data, sr)
                            else:
                                st.error("Não foi possível extrair features do áudio.")
                                logging.warning("Não foi possível extrair features do áudio.")
                        elif metodo_classificacao == "ResNet-18":
                            try:
                                espectrograma = gerar_espectrograma(data, sr)
                                if espectrograma:
                                    transform = torch_transforms.Compose([
                                        torch_transforms.Resize((224, 224)),
                                        torch_transforms.ToTensor(),
                                        torch_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
                                    ])
                                    espectrograma = transform(espectrograma).unsqueeze(0)
                                    modelo.eval()
                                    modelo = modelo.to('cuda' if torch.cuda.is_available() else 'cpu')
                                    with torch.no_grad():
                                        espectrograma = espectrograma.to('cuda' if torch.cuda.is_available() else 'cpu')
                                        pred = modelo(espectrograma)
                                        probs = torch.nn.functional.softmax(pred, dim=1)
                                        confidence, pred_class = torch.max(probs, dim=1)
                                        pred_label = classes[pred_class.item()]
                                        confidence = confidence.item() * 100
                                        st.markdown(f"**Classe Predita:** {pred_label} (Confiança: {confidence:.2f}%)")

                                        fig_prob, ax_prob = plt.subplots(figsize=(8,4))
                                        ax_prob.bar(classes, probs.cpu().numpy()[0], color='skyblue')
                                        ax_prob.set_title("Probabilidades por Classe")
                                        ax_prob.set_ylabel("Probabilidade")
                                        plt.xticks(rotation=45)
                                        st.pyplot(fig_prob)
                                        plt.close(fig_prob)

                                        st.audio(caminho_audio)
                                        visualizar_audio(data, sr)

                                        espectrograma_img = espectrograma.cpu().squeeze().permute(1,2,0).numpy()
                                        espectrograma_img = np.clip(espectrograma_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
                                        st.image(espectrograma_img, caption="Espectrograma Classificado", use_column_width=True)
                            except Exception as e:
                                st.error(f"Erro na predição ResNet-18: {e}")
                                logging.error(f"Erro na predição ResNet-18: {e}")
                    else:
                        st.error("Não foi possível carregar o áudio.")
                        logging.warning("Não foi possível carregar o áudio.")
                except Exception as e:
                    st.error(f"Erro ao processar o áudio: {e}")
                    logging.error(f"Erro ao processar o áudio: {e}")

def treinar_modelo(SEED):
    """
    Função para treinar o modelo CNN ou ResNet-18.

    Args:
        SEED (int): Valor da semente para reprodutibilidade.
    """
    with st.expander("Treinamento do Modelo CNN ou ResNet-18"):
        metodo_treinamento = st.selectbox(
            "Escolha o Método de Treinamento:",
            ["CNN Personalizada", "ResNet-18"],
            key='select_training_method'
        )

        # Controle para parar o treinamento
        stop_training_choice = st.sidebar.checkbox("Permitir Parar Treinamento a Qualquer Momento", value=False, key='stop_training_checkbox')

        if stop_training_choice:
            st.sidebar.write("Durante o treinamento, caso deseje parar, clique no botão abaixo:")
            stop_button = st.sidebar.button("Parar Treinamento Agora", key='stop_training_button')
            if stop_button:
                st.session_state.stop_training = True
        else:
            st.session_state.stop_training = False

        st.markdown("### Passo 1: Upload do Dataset (ZIP)")
        zip_upload = st.file_uploader("Upload do ZIP", type=["zip"], key='zip_uploader')

        if zip_upload is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                    tmp_zip.write(zip_upload.read())
                    caminho_zip = tmp_zip.name

                diretorio_extracao, categorias = carregar_dados(caminho_zip)
                if diretorio_extracao is None:
                    st.error("Falha ao extrair o dataset.")
                    return

                if len(categorias) == 0:
                    st.error("Nenhuma subpasta encontrada no ZIP.")
                    return

                st.success("Dataset extraído!")
                st.write(f"Classes encontradas: {', '.join(categorias)}")

                caminhos_arquivos = []
                labels = []
                for cat in categorias:
                    caminho_cat = os.path.join(diretorio_extracao, cat)
                    arquivos_na_cat = [f for f in os.listdir(caminho_cat) if f.lower().endswith(('.wav','.mp3','.flac','.ogg','.m4a'))]
                    st.write(f"Classe '{cat}': {len(arquivos_na_cat)} arquivos.")
                    for nome_arquivo in arquivos_na_cat:
                        caminhos_arquivos.append(os.path.join(caminho_cat, nome_arquivo))
                        labels.append(cat)

                df = pd.DataFrame({'caminho_arquivo': caminhos_arquivos, 'classe': labels})
                st.write("10 Primeiras Amostras do Dataset:")
                st.dataframe(df.head(10))

                if len(df) == 0:
                    st.error("Nenhuma amostra encontrada no dataset.")
                    return

                labelencoder = LabelEncoder()
                y = labelencoder.fit_transform(df['classe'])
                classes = labelencoder.classes_

                st.write(f"Classes codificadas: {', '.join(classes)}")

                X_aug = []
                y_aug = []

                # Extrair features ou gerar espectrogramas dependendo do método de treinamento
                if metodo_treinamento == "CNN Personalizada":
                    st.write("Extraindo Features (MFCCs, Centróide)...")
                    X = []
                    y_valid = []
                    for i, row in df.iterrows():
                        if st.session_state.stop_training:
                            break
                        arquivo = row['caminho_arquivo']
                        data, sr = carregar_audio(arquivo, sr=None)
                        if data is not None:
                            ftrs = extrair_features(data, sr, use_mfcc=True, use_spectral_centroid=True)
                            if ftrs is not None:
                                X.append(ftrs)
                                y_valid.append(y[i])
                    X = np.array(X)
                    y_valid = np.array(y_valid)
                elif metodo_treinamento == "ResNet-18":
                    st.write("Gerando Espectrogramas...")
                    X = []
                    y_valid = []
                    for i, row in df.iterrows():
                        if st.session_state.stop_training:
                            break
                        arquivo = row['caminho_arquivo']
                        data, sr = carregar_audio(arquivo, sr=None)
                        if data is not None:
                            espectrograma = gerar_espectrograma(data, sr)
                            if espectrograma is not None:
                                temp_dir = tempfile.mkdtemp()
                                img_path = os.path.join(temp_dir, f"class_{y[i]}_img_{i}.png")
                                espectrograma.save(img_path)
                                X.append(img_path)
                                y_valid.append(y[i])

                st.write(f"Dados Processados: {len(X)} amostras.")

                st.sidebar.markdown("**Configurações de Treinamento:**")

                # Configurações de treinamento
                num_epochs = st.sidebar.slider(
                    "Número de Épocas:",
                    10, 500, 50, 10,
                    key='num_epochs_training'
                )
                batch_size = st.sidebar.selectbox(
                    "Batch:",
                    [8,16,32,64,128],0,
                    key='batch_size_training'
                )

                treino_percentage = st.sidebar.slider(
                    "Treino (%)",
                    50, 90, 70, 5,
                    key='treino_percentage_training'
                )
                valid_percentage = st.sidebar.slider(
                    "Validação (%)",
                    5, 30, 15, 5,
                    key='valid_percentage_training'
                )
                test_percentage = 100 - (treino_percentage + valid_percentage)
                if test_percentage < 0:
                    st.sidebar.error("Treino + Validação > 100%")
                    st.stop()
                st.sidebar.write(f"Teste (%)={test_percentage}%")

                augment_factor = st.sidebar.slider(
                    "Fator Aumento:",
                    1, 100, 10, 1,
                    key='augment_factor_training'
                )
                dropout_rate = st.sidebar.slider(
                    "Dropout:",
                    0.0, 0.9, 0.4, 0.05,
                    key='dropout_rate_training'
                )

                regularization_type = st.sidebar.selectbox(
                    "Regularização:",
                    ["None","L1","L2","L1_L2"],0,
                    key='regularization_type_training'
                )
                if regularization_type == "L1":
                    l1_regularization = st.sidebar.slider(
                        "L1:",0.0,0.1,0.001,0.001,
                        key='l1_regularization_training'
                    )
                    l2_regularization = 0.0
                elif regularization_type == "L2":
                    l2_regularization = st.sidebar.slider(
                        "L2:",0.0,0.1,0.001,0.001,
                        key='l2_regularization_training'
                    )
                    l1_regularization = 0.0
                elif regularization_type == "L1_L2":
                    l1_regularization = st.sidebar.slider(
                        "L1:",0.0,0.1,0.001,0.001,
                        key='l1_regularization_training'
                    )
                    l2_regularization = st.sidebar.slider(
                        "L2:",0.0,0.1,0.001,0.001,
                        key='l2_regularization_training'
                    )
                else:
                    l1_regularization = 0.0
                    l2_regularization = 0.0

                st.sidebar.markdown("**Fine-Tuning Adicional:**")
                learning_rate = st.sidebar.slider(
                    "Taxa de Aprendizado:",
                    1e-5, 1e-2, 1e-3, step=1e-5, format="%.5f",
                    key='learning_rate_training'
                )
                optimizer_choice = st.sidebar.selectbox(
                    "Otimização:",
                    ["Adam", "SGD", "RMSprop"],0,
                    key='optimizer_choice_training'
                )

                enable_augmentation = st.sidebar.checkbox(
                    "Data Augmentation", True,
                    key='enable_augmentation_training'
                )
                if enable_augmentation:
                    adicionar_ruido = st.sidebar.checkbox("Ruído Gaussiano", True, key='add_gaussian_noise_training')
                    estiramento_tempo = st.sidebar.checkbox("Time Stretch", True, key='time_stretch_training')
                    alteracao_pitch = st.sidebar.checkbox("Pitch Shift", True, key='pitch_shift_training')
                    deslocamento = st.sidebar.checkbox("Deslocamento", True, key='shift_training')

                cross_validation = st.sidebar.checkbox(
                    "k-Fold?", False,
                    key='cross_validation_training'
                )
                if cross_validation:
                    k_folds = st.sidebar.number_input(
                        "Folds:",2,10,5,1,
                        key='k_folds_training'
                    )
                else:
                    k_folds = 1

                balance_classes = st.sidebar.selectbox(
                    "Balanceamento:",["Balanced","None"],0,
                    key='balance_classes_training'
                )

                # Aumentação de dados
                if metodo_treinamento == "CNN Personalizada" and enable_augmentation:
                    st.write("Aumentando Dados para CNN Personalizada...")
                    transforms_aug = []
                    if adicionar_ruido:
                        transforms_aug.append(AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0))
                    if estiramento_tempo:
                        transforms_aug.append(TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0))
                    if alteracao_pitch:
                        transforms_aug.append(PitchShift(min_semitones=-4, max_semitones=4, p=1.0))
                    if deslocamento:
                        transforms_aug.append(Shift(min_shift=-0.5, max_shift=0.5, p=1.0))
                    if transforms_aug:
                        aug = Compose(transforms_aug)
                        for i, row in df.iterrows():
                            if st.session_state.stop_training:
                                break
                            arquivo = row['caminho_arquivo']
                            data, sr = carregar_audio(arquivo, sr=None)
                            if data is not None:
                                for _ in range(augment_factor):
                                    if st.session_state.stop_training:
                                        break
                                    aug_data = aumentar_audio(data, sr, aug)
                                    ftrs = extrair_features(aug_data, sr, use_mfcc=True, use_spectral_centroid=True)
                                    if ftrs is not None:
                                        X_aug.append(ftrs)
                                        y_aug.append(y[i])

                if metodo_treinamento == "ResNet-18" and enable_augmentation:
                    st.write("Aumentando Dados para ResNet-18...")
                    transforms_aug_resnet = Compose([
                        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                        Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
                    ])
                    for i, row in df.iterrows():
                        if st.session_state.stop_training:
                            break
                        arquivo = row['caminho_arquivo']
                        data, sr = carregar_audio(arquivo, sr=None)
                        if data is not None:
                            aug_data = aumentar_audio(data, sr, transforms_aug_resnet)
                            espectrograma_aug = gerar_espectrograma(aug_data, sr)
                            if espectrograma_aug:
                                temp_dir = tempfile.mkdtemp()
                                img_path = os.path.join(temp_dir, f"class_{y[i]}_aug_img_{i}.png")
                                espectrograma_aug.save(img_path)
                                X_aug.append(img_path)
                                y_aug.append(y[i])

                # Combinar dados originais e aumentados
                if metodo_treinamento == "CNN Personalizada" and enable_augmentation and len(X_aug) > 0 and len(y_aug) > 0:
                    X_aug = np.array(X_aug)
                    y_aug = np.array(y_aug)
                    st.write(f"Dados Aumentados: {X_aug.shape}")
                    X_combined = np.concatenate((X, X_aug), axis=0)
                    y_combined = np.concatenate((y_valid, y_aug), axis=0)
                elif metodo_treinamento == "ResNet-18" and enable_augmentation and len(X_aug) > 0 and len(y_aug) > 0:
                    X_aug = np.array(X_aug)
                    y_aug = np.array(y_aug)
                    st.write(f"Dados Aumentados: {X_aug.shape}")
                    X_combined = np.concatenate((X, X_aug), axis=0)
                    y_combined = np.concatenate((y_valid, y_aug), axis=0)
                else:
                    X_combined = X
                    y_combined = y_valid

                # Verificação de amostras válidas para ResNet-18
                if metodo_treinamento == "ResNet-18":
                    valid_indices = []
                    for i, img_path in enumerate(X_combined):
                        if os.path.exists(img_path):
                            valid_indices.append(i)
                        else:
                            logging.warning(f"Arquivo de espectrograma inexistente: {img_path}")
                    X_combined = X_combined[valid_indices]
                    y_combined = y_combined[valid_indices]
                    st.write(f"Amostras válidas após verificação: {len(X_combined)}")

                # Verificar se todas as classes têm pelo menos k_folds amostras
                if not verificar_contagem_classes(y_combined, k_folds=k_folds):
                    st.error("Algumas classes têm menos amostras do que o número de folds.")
                    st.stop()

                st.write("Dividindo Dados...")
                if metodo_treinamento == "CNN Personalizada":
                    X_train, X_temp, y_train, y_temp = train_test_split(
                        X_combined, y_combined, 
                        test_size=(100 - treino_percentage)/100.0,
                        random_state=SEED, 
                        stratify=y_combined
                    )
                elif metodo_treinamento == "ResNet-18":
                    X_train, X_temp, y_train, y_temp = train_test_split(
                        X_combined, y_combined, 
                        test_size=(100 - treino_percentage)/100.0,
                        random_state=SEED, 
                        stratify=y_combined
                    )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, 
                    test_size=test_percentage/(test_percentage + valid_percentage),
                    random_state=SEED, 
                    stratify=y_temp
                )

                st.write(f"Treino: {len(X_train)}, Validação: {len(X_val)}, Teste: {len(X_test)}")

                if metodo_treinamento == "ResNet-18":
                    transform_train = torch_transforms.Compose([
                        torch_transforms.Resize((224, 224)),
                        torch_transforms.RandomHorizontalFlip(),
                        torch_transforms.ToTensor(),
                        torch_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                    ])
                    transform_val = torch_transforms.Compose([
                        torch_transforms.Resize((224, 224)),
                        torch_transforms.ToTensor(),
                        torch_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                    ])

                    dataset_train = AudioSpectrogramDataset(
                        pd.DataFrame({
                            'caminho_arquivo': X_train,
                            'classe': [classes[i] for i in y_train]
                        }), 
                        classes, 
                        transform=transform_train
                    )
                    dataset_val = AudioSpectrogramDataset(
                        pd.DataFrame({
                            'caminho_arquivo': X_val,
                            'classe': [classes[i] for i in y_val]
                        }), 
                        classes, 
                        transform=transform_val
                    )
                    dataset_test = AudioSpectrogramDataset(
                        pd.DataFrame({
                            'caminho_arquivo': X_test,
                            'classe': [classes[i] for i in y_test]
                        }), 
                        classes, 
                        transform=transform_val
                    )

                    # Use num_workers=0 para facilitar a depuração
                    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate)
                    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate)
                    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate)
                else:
                    X_train_final = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                    X_val_final = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
                    X_test_final = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

                st.sidebar.markdown("**Configurações de Treinamento Adicionais:**")

                if balance_classes == "Balanced":
                    if metodo_treinamento == "CNN Personalizada":
                        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
                    elif metodo_treinamento == "ResNet-18":
                        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                        class_weight = torch.tensor(class_weights, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')
                else:
                    class_weight_dict = None
                    class_weight = None

                if metodo_treinamento == "CNN Personalizada":
                    num_conv_layers = st.sidebar.slider(
                        "Conv Layers",
                        1, 5, 2, 1,
                        key='num_conv_layers_cnn'
                    )
                    conv_filters_str = st.sidebar.text_input(
                        "Filtros (vírgula):","64,128",
                        key='conv_filters_cnn'
                    )
                    conv_kernel_size_str = st.sidebar.text_input(
                        "Kernel (vírgula):","10,10",
                        key='conv_kernel_size_cnn'
                    )
                    conv_filters = [int(f.strip()) for f in conv_filters_str.split(',')]
                    conv_kernel_size = [int(k.strip()) for k in conv_kernel_size_str.split(',')]

                    input_length = X_train_final.shape[1]
                    for i in range(num_conv_layers):
                        if conv_kernel_size[i] > input_length:
                            conv_kernel_size[i] = input_length

                    num_dense_layers = st.sidebar.slider(
                        "Dense Layers:",
                        1, 3, 1, 1,
                        key='num_dense_layers_cnn'
                    )
                    dense_units_str = st.sidebar.text_input(
                        "Neurônios Dense (vírgula):","64",
                        key='dense_units_cnn'
                    )
                    dense_units = [int(u.strip()) for u in dense_units_str.split(',')]
                    if len(dense_units) != num_dense_layers:
                        st.sidebar.error("Número de neurônios deve ser igual ao número de camadas Dense.")
                        st.stop()

                    if regularization_type == "L1":
                        reg = regularizers.l1(l1_regularization)
                    elif regularization_type == "L2":
                        reg = regularizers.l2(l2_regularization)
                    elif regularization_type == "L1_L2":
                        reg = regularizers.l1_l2(l1=l1_regularization, l2=l2_regularization)
                    else:
                        reg = None

                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input

                    modelo_cnn = Sequential()
                    modelo_cnn.add(Input(shape=(X_train_final.shape[1],1)))
                    for i in range(num_conv_layers):
                        modelo_cnn.add(Conv1D(
                            filters=conv_filters[i],
                            kernel_size=conv_kernel_size[i],
                            activation='relu',
                            kernel_regularizer=reg
                        ))
                        modelo_cnn.add(Dropout(dropout_rate))
                        modelo_cnn.add(MaxPooling1D(pool_size=2))

                    modelo_cnn.add(Flatten())
                    for i in range(num_dense_layers):
                        modelo_cnn.add(Dense(
                            units=dense_units[i],
                            activation='relu',
                            kernel_regularizer=reg
                        ))
                        modelo_cnn.add(Dropout(dropout_rate))
                    modelo_cnn.add(Dense(len(classes), activation='softmax'))

                    if optimizer_choice == "Adam":
                        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                    elif optimizer_choice == "SGD":
                        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
                    elif optimizer_choice == "RMSprop":
                        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
                    else:
                        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

                    modelo_cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

                elif metodo_treinamento == "ResNet-18":
                    # Definindo a função de perda e otimizador
                    criterion = torch.nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)

                st.sidebar.markdown("**Configurações de Treinamento Adicionais:**")

                es_monitor = st.sidebar.selectbox(
                    "Monitor (Early Stopping):", 
                    ["val_loss","val_accuracy"],
                    key='es_monitor'
                )
                es_patience = st.sidebar.slider(
                    "Patience:",
                    1, 20, 5, 1,
                    key='es_patience'
                )
                es_mode = st.sidebar.selectbox(
                    "Mode:",
                    ["min","max"],
                    key='es_mode'
                )

                # Configurações de callbacks
                if metodo_treinamento == "CNN Personalizada":
                    try:
                        checkpointer = ModelCheckpoint(
                            os.path.join('modelos_salvos','modelo_agua_aumentado.keras'),
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True
                        )
                        earlystop = EarlyStopping(
                            monitor=es_monitor,
                            patience=es_patience,
                            restore_best_weights=True,
                            mode=es_mode,
                            verbose=1
                        )
                        lr_scheduler = ReduceLROnPlateau(
                            monitor='val_loss', 
                            factor=0.5, 
                            patience=3, 
                            verbose=1
                        )
                        callbacks = [checkpointer, earlystop, lr_scheduler]
                    except Exception as e:
                        st.error(f"Erro ao configurar callbacks: {e}")
                        logging.error(f"Erro ao configurar callbacks: {e}")
                        return
                elif metodo_treinamento == "ResNet-18":
                    # PyTorch não usa callbacks da mesma forma que Keras, mas você pode implementar funcionalidades similares
                    callbacks = []

                st.write("Treinando...")
                with st.spinner('Treinando...'):
                    if metodo_treinamento == "CNN Personalizada":
                        try:
                            if cross_validation and k_folds > 1:
                                kf = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)
                                fold_no = 1
                                val_scores = []
                                for train_index, val_index in kf.split(X_train_final):
                                    if st.session_state.stop_training:
                                        st.warning("Treinamento Parado pelo Usuário!")
                                        break
                                    st.write(f"Fold {fold_no}")
                                    X_train_cv, X_val_cv = X_train_final[train_index], X_train_final[val_index]
                                    y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]
                                    historico = modelo_cnn.fit(
                                        X_train_cv, to_categorical(y_train_cv),
                                        epochs=num_epochs,
                                        batch_size=batch_size,
                                        validation_data=(X_val_cv, to_categorical(y_val_cv)),
                                        callbacks=callbacks,
                                        class_weight=class_weight_dict,
                                        verbose=1
                                    )
                                    score = modelo_cnn.evaluate(X_val_cv, to_categorical(y_val_cv), verbose=0)
                                    val_scores.append(score[1]*100)
                                    fold_no += 1
                                if not st.session_state.stop_training:
                                    st.write(f"Acurácia Média CV: {np.mean(val_scores):.2f}%")
                            else:
                                historico = modelo_cnn.fit(
                                    X_train_final, to_categorical(y_train),
                                    epochs=num_epochs,
                                    batch_size=batch_size,
                                    validation_data=(X_val_final, to_categorical(y_val)),
                                    callbacks=callbacks,
                                    class_weight=class_weight_dict,
                                    verbose=1
                                )
                                if st.session_state.stop_training:
                                    st.warning("Treinamento Parado pelo Usuário!")
                                else:
                                    st.success("Treino concluído!")
                        except Exception as e:
                            st.error(f"Erro durante o treinamento da CNN: {e}")
                            logging.error(f"Erro durante o treinamento da CNN: {e}")

                        try:
                            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_model:
                                modelo_cnn.save(tmp_model.name)
                                caminho_tmp_model = tmp_model.name
                            with open(caminho_tmp_model, 'rb') as f:
                                modelo_bytes = f.read()
                            buffer = io.BytesIO(modelo_bytes)
                            st.download_button("Download Modelo (.keras)", data=buffer, file_name="modelo_agua_aumentado.keras")
                            os.remove(caminho_tmp_model)
                        except Exception as e:
                            st.error(f"Erro ao salvar o modelo: {e}")
                            logging.error(f"Erro ao salvar o modelo: {e}")

                        try:
                            classes_str = "\n".join(classes)
                            st.download_button("Download Classes (classes.txt)", data=classes_str, file_name="classes.txt")
                        except Exception as e:
                            st.error(f"Erro ao salvar o arquivo de classes: {e}")
                            logging.error(f"Erro ao salvar o arquivo de classes: {e}")

                        if not st.session_state.stop_training:
                            try:
                                st.markdown("### Avaliação do Modelo")
                                score_train = modelo_cnn.evaluate(X_train_final, to_categorical(y_train), verbose=0)
                                score_val = modelo_cnn.evaluate(X_val_final, to_categorical(y_val), verbose=0)
                                score_test = modelo_cnn.evaluate(X_test_final, to_categorical(y_test), verbose=0)

                                st.write(f"Acurácia Treino: {score_train[1]*100:.2f}%")
                                st.write(f"Acurácia Validação: {score_val[1]*100:.2f}%")
                                st.write(f"Acurácia Teste: {score_test[1]*100:.2f}%")

                                y_pred = modelo_cnn.predict(X_test_final)
                                y_pred_classes = np.argmax(y_pred, axis=1)
                                f1_val = f1_score(y_test, y_pred_classes, average='weighted', zero_division=0)
                                prec = precision_score(y_test, y_pred_classes, average='weighted', zero_division=0)
                                rec = recall_score(y_test, y_pred_classes, average='weighted', zero_division=0)

                                st.write(f"F1-score: {f1_val*100:.2f}%")
                                st.write(f"Precisão: {prec*100:.2f}%")
                                st.write(f"Recall: {rec*100:.2f}%")

                                st.markdown("### Matriz de Confusão")
                                cm = confusion_matrix(y_test, y_pred_classes, labels=range(len(classes)))
                                cm_df = pd.DataFrame(cm, index=classes, columns=classes)
                                fig_cm, ax_cm = plt.subplots(figsize=(12,8))
                                sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                ax_cm.set_title("Matriz de Confusão", fontsize=16)
                                ax_cm.set_xlabel("Classe Prevista", fontsize=14)
                                ax_cm.set_ylabel("Classe Real", fontsize=14)
                                st.pyplot(fig_cm)
                                plt.close(fig_cm)

                                st.markdown("### Histórico de Treinamento")
                                hist_df = pd.DataFrame(historico.history)
                                st.dataframe(hist_df)

                                fig_hist, (ax_loss, ax_acc) = plt.subplots(1,2, figsize=(12,4))
                                ax_loss.plot(hist_df.index, hist_df['loss'], label='Treino')
                                ax_loss.plot(hist_df.index, hist_df['val_loss'], label='Validação')
                                ax_loss.set_title("Curva de Perda")
                                ax_loss.set_xlabel("Épocas")
                                ax_loss.set_ylabel("Loss")
                                ax_loss.legend()

                                ax_acc.plot(hist_df.index, hist_df['accuracy'], label='Treino')
                                ax_acc.plot(hist_df.index, hist_df['val_accuracy'], label='Validação')
                                ax_acc.set_title("Curva de Acurácia")
                                ax_acc.set_xlabel("Épocas")
                                ax_acc.set_ylabel("Acurácia")
                                ax_acc.legend()

                                st.pyplot(fig_hist)
                                plt.close(fig_hist)

                                st.markdown("### Explicabilidade com SHAP")
                                st.write("Selecionando amostras de teste para análise SHAP.")
                                X_sample = X_test_final[:50]
                                try:
                                    explainer = shap.DeepExplainer(modelo_cnn, X_train_final[:100])
                                    shap_values = explainer.shap_values(X_sample)

                                    st.write("Plot SHAP Summary por Classe:")

                                    num_shap_values = len(shap_values)
                                    num_classes = len(classes)

                                    if num_shap_values == num_classes:
                                        for class_idx, class_name in enumerate(classes):
                                            st.write(f"**Classe: {class_name}**")
                                            fig_shap = plt.figure()
                                            shap.summary_plot(shap_values[class_idx], X_sample, show=False)
                                            st.pyplot(fig_shap)
                                            plt.close(fig_shap)
                                    elif num_shap_values == 1 and num_classes == 2:
                                        st.write(f"**Classe: {classes[1]}**")
                                        fig_shap = plt.figure()
                                        shap.summary_plot(shap_values[0], X_sample, show=False)
                                        st.pyplot(fig_shap)
                                        plt.close(fig_shap)
                                    else:
                                        st.warning("Número de shap_values não corresponde ao número de classes.")
                                        st.write(f"shap_values length: {num_shap_values}, classes length: {num_classes}")
                                    st.write("""
                                    **Interpretação SHAP:**  
                                    MFCCs com valor SHAP alto contribuem significativamente para a classe.  
                                    Frequências associadas a modos ressonantes específicos tornam certas classes mais prováveis.
                                    """)
                                except Exception as e:
                                    st.write("SHAP não pôde ser gerado:", e)
                                    logging.error(f"Erro ao gerar SHAP: {e}")

                                st.markdown("### Análise de Clusters (K-Means e Hierárquico)")
                                st.write("""
                                Clustering revela como dados se agrupam.  
                                Determinaremos k automaticamente usando o coeficiente de silhueta.
                                """)

                                melhor_k = escolher_k_kmeans(X_combined, y_combined, max_k=10)
                                sil_score = silhouette_score(X_combined, KMeans(n_clusters=melhor_k, random_state=42).fit_predict(X_combined))
                                st.write(f"Melhor k encontrado para K-Means: {melhor_k} (Silhueta={sil_score:.2f})")

                                kmeans = KMeans(n_clusters=melhor_k, random_state=42)
                                kmeans_labels = kmeans.fit_predict(X_combined)
                                st.write("Classes por Cluster (K-Means):")
                                cluster_dist = []
                                for cidx in range(melhor_k):
                                    cluster_classes = y_combined[kmeans_labels == cidx]
                                    counts = pd.Series(cluster_classes).value_counts()
                                    cluster_dist.append(counts)
                                for idx, dist in enumerate(cluster_dist):
                                    st.write(f"**Cluster {idx+1}:**")
                                    st.write(dist)

                                st.write("Análise Hierárquica:")
                                Z = linkage(X_combined, 'ward')
                                fig_dend, ax_dend = plt.subplots(figsize=(10,5))
                                dendrogram(Z, ax=ax_dend, truncate_mode='level', p=5)
                                ax_dend.set_title("Dendrograma Hierárquico")
                                ax_dend.set_xlabel("Amostras")
                                ax_dend.set_ylabel("Distância")
                                st.pyplot(fig_dend)
                                plt.close(fig_dend)

                                hier = AgglomerativeClustering(n_clusters=2)
                                hier_labels = hier.fit_predict(X_combined)
                                st.write("Classes por Cluster (Hierárquico):")
                                cluster_dist_h = []
                                for cidx in range(2):
                                    cluster_classes = y_combined[hier_labels == cidx]
                                    counts_h = pd.Series(cluster_classes).value_counts()
                                    cluster_dist_h.append(counts_h)
                                for idx, dist in enumerate(cluster_dist_h):
                                    st.write(f"**Cluster {idx+1}:**")
                                    st.write(dist)

                                st.markdown("### Visualização de Exemplos")
                                visualizar_exemplos_classe(df, y_valid, classes, augmentation=enable_augmentation, sr=22050, metodo=metodo_treinamento)
                            except Exception as e:
                                st.error(f"Erro durante a avaliação do modelo: {e}")
                                logging.error(f"Erro durante a avaliação do modelo: {e}")

                    elif metodo_treinamento == "ResNet-18":
                        # Implementação semelhante para ResNet-18
                        try:
                            # Treinamento usando PyTorch
                            modelo.train()
                            for epoch in range(num_epochs):
                                if st.session_state.stop_training:
                                    st.warning("Treinamento Parado pelo Usuário!")
                                    break
                                running_loss = 0.0
                                for inputs, labels in loader_train:
                                    inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                                    optimizer.zero_grad()
                                    outputs = modelo(inputs)
                                    loss = criterion(outputs, labels)
                                    loss.backward()
                                    optimizer.step()
                                    running_loss += loss.item()
                                avg_loss = running_loss / len(loader_train)
                                st.write(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

                                # Validação
                                modelo.eval()
                                val_loss = 0.0
                                correct = 0
                                total = 0
                                with torch.no_grad():
                                    for inputs, labels in loader_val:
                                        inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                                        outputs = modelo(inputs)
                                        loss = criterion(outputs, labels)
                                        val_loss += loss.item()
                                        _, predicted = torch.max(outputs.data, 1)
                                        total += labels.size(0)
                                        correct += (predicted == labels).sum().item()
                                avg_val_loss = val_loss / len(loader_val)
                                val_accuracy = 100 * correct / total
                                st.write(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
                                modelo.train()

                                if st.session_state.stop_training:
                                    st.warning("Treinamento Parado pelo Usuário!")
                                    break

                            st.success("Treino concluído!")

                            # Avaliação no Teste
                            modelo.eval()
                            test_loss = 0.0
                            correct = 0
                            total = 0
                            with torch.no_grad():
                                for inputs, labels in loader_test:
                                    inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                                    outputs = modelo(inputs)
                                    loss = criterion(outputs, labels)
                                    test_loss += loss.item()
                                    _, predicted = torch.max(outputs.data, 1)
                                    total += labels.size(0)
                                    correct += (predicted == labels).sum().item()
                            avg_test_loss = test_loss / len(loader_test)
                            test_accuracy = 100 * correct / total
                            st.write(f"Acurácia Teste: {test_accuracy:.2f}%")
                        except Exception as e:
                            st.error(f"Erro durante o treinamento da ResNet-18: {e}")
                            logging.error(f"Erro durante o treinamento da ResNet-18: {e}")

            except Exception as e:
                st.error(f"Erro ao treinar o modelo: {e}")
                logging.error(f"Erro ao treinar o modelo: {e}")

            # Limpeza de arquivos temporários
            try:
                gc.collect()
                os.remove(caminho_zip)
                for cat in categorias:
                    caminho_cat = os.path.join(diretorio_extracao, cat)
                    for arquivo in os.listdir(caminho_cat):
                        os.remove(os.path.join(caminho_cat, arquivo))
                    os.rmdir(caminho_cat)
                os.rmdir(diretorio_extracao)
                logging.info("Processo concluído.")
            except Exception as e:
                logging.error(f"Erro ao limpar arquivos temporários: {e}")

# Interface do Streamlit
def main():
    """
    Função principal que define a interface do Streamlit.
    """
    st.title("Classificador de Sons de Água em Copos de Vidro")
    st.write("Bem-vindo ao Classificador de Sons de Água! Aqui você pode treinar modelos para reconhecer diferentes sons de água em copos de vidro ou usar um modelo treinado para classificar novos áudios.")

    app_mode = st.sidebar.radio("Escolha a seção", ["Classificar Áudio", "Treinar Modelo"], key='app_mode')

    # Configurações Gerais
    st.sidebar.header("Configurações Gerais")
    with st.sidebar.expander("Parâmetro SEED e Reprodutibilidade"):
        st.markdown("**SEED** garante resultados reproduzíveis.")
    
    seed_selection = st.sidebar.selectbox(
        "Escolha o valor do SEED:",
        options=list(range(0, 61, 2)),
        index=list(range(0, 61, 2)).index(42),
        help="Define a semente para reprodutibilidade.",
        key='seed_selection'
    )
    SEED = seed_selection
    set_seeds(SEED)

    with st.sidebar.expander("Sobre o SEED"):
        st.markdown("""
        **SEED** garante replicabilidade de resultados, permitindo que os experimentos sejam reproduzidos com os mesmos dados e parâmetros.
        """)

    eu_icon_path = "eu.ico"
    if os.path.exists(eu_icon_path):
        try:
            st.sidebar.image(eu_icon_path, width=80)
        except UnidentifiedImageError:
            st.sidebar.text("Ícone 'eu.ico' corrompido.")
    else:
        st.sidebar.text("Ícone 'eu.ico' não encontrado.")

    st.sidebar.write("Desenvolvido por Projeto Geomaker + IA")

    if app_mode == "Classificar Áudio":
        classificar_audio(SEED)
    elif app_mode == "Treinar Modelo":
        treinar_modelo(SEED)

    with st.expander("Contexto e Descrição Completa"):
        st.markdown("""
        **Classificação de Sons de Água Vibrando em Copo de Vidro com Aumento de Dados, CNN Personalizada e ResNet-18**

        Este aplicativo realiza duas tarefas principais:

        1. **Treinar Modelo:**  
           - Você faz upload de um dataset .zip contendo pastas, cada pasta representando uma classe (estado físico do fluido-copo).
           - Escolhe entre treinar uma CNN personalizada ou utilizar a ResNet-18 pré-treinada.
           - O app extrai características do áudio (MFCCs, centróide espectral ou espectrogramas), normaliza, aplica (opcionalmente) Data Augmentation.
           - Treina o modelo escolhido para classificar os sons.
           - Mostra métricas (acurácia, F1, precisão, recall) e histórico de treinamento, bem como gráficos das curvas de perda e acurácia.
           - Plota a Matriz de Confusão, permitindo visualizar onde o modelo se confunde.
           - Utiliza SHAP para interpretar quais features são mais importantes (apenas para CNN personalizada).
           - Executa clustering (K-Means e Hierárquico) para entender a distribuição interna dos dados, exibindo o dendrograma.
           - Implementa LR Scheduler (ReduceLROnPlateau) para refinar o treinamento.
           - Possibilita visualizar gráficos de espectro (frequência x amplitude), espectrogramas e MFCCs.
           - Mostra exemplos de cada classe do dataset original e exemplos aumentados.

        2. **Classificar Áudio com Modelo Treinado:**  
           - Você faz upload de um modelo já treinado (.keras, .h5 para CNN personalizada ou .pth para ResNet-18) e do arquivo de classes (classes.txt).
           - Escolhe o método de classificação utilizado no treinamento.
           - Envia um arquivo de áudio para classificação.
           - O app extrai as mesmas features ou gera espectrogramas e prediz a classe do áudio, mostrando probabilidades e um gráfico de barras das probabilidades.
           - Possibilidade de visualizar o espectro do áudio classificado (FFT), forma de onda, espectrograma e MFCCs ou espectrograma correspondente.

        **Contexto Físico (Fluidos, Ondas, Calor):**
        Ao perturbar um copo com água, surgem modos ressonantes. A quantidade de água e a temperatura influenciam as frequências ressonantes. As MFCCs e centróide refletem a distribuição espectral, e os espectrogramas capturam a variação tempo-frequência do som. A CNN personalizada ou a ResNet-18 aprendem padrões ligados ao estado do fluido-copo.

        **Explicação para Leigos:**
        Imagine o copo como um instrumento: menos água = som mais agudo; mais água = som mais grave. O computador converte o som em números (MFCCs, centróide ou espectrogramas), a CNN ou ResNet aprende a relacioná-los à quantidade de água e outras condições. SHAP explica quais características importam, clustering mostra agrupamentos de sons. Visualizações tornam tudo compreensível.

        Em suma, este app integra teoria física, processamento de áudio, machine learning, interpretabilidade e análise exploratória de dados, proporcionando uma ferramenta poderosa e intuitiva para classificação de sons de água em copos de vidro.
        """)

if __name__ == "__main__":
    main()

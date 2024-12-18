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
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

# Configuração de Logging
logging.basicConfig(
    filename='experiment_logs.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Configurações da Página
seed_options = list(range(0, 61, 2))
default_seed = 42
if default_seed not in seed_options:
    seed_options.insert(0, default_seed)

icon_path = "logo.png"
if os.path.exists(icon_path):
    try:
        st.set_page_config(page_title="Geomaker", page_icon=icon_path, layout="wide")
        logging.info("Ícone carregado com sucesso.")
    except Exception as e:
        st.set_page_config(page_title="Geomaker", layout="wide")
        logging.warning(f"Erro ao carregar ícone: {e}")
else:
    st.set_page_config(page_title="Geomaker", layout="wide")
    logging.warning("Ícone não encontrado.")

# Estado para parar o treinamento
if 'stop_training' not in st.session_state:
    st.session_state.stop_training = False

def set_seeds(seed):
    """Define as sementes para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def carregar_audio(caminho_arquivo, sr=None):
    """Carrega o arquivo de áudio usando Librosa."""
    try:
        data, sr = librosa.load(caminho_arquivo, sr=sr, res_type='kaiser_fast')
        return data, sr
    except Exception as e:
        logging.error(f"Erro ao carregar áudio: {e}")
        return None, None

def extrair_features(data, sr, use_mfcc=True, use_spectral_centroid=True):
    """
    Extrai MFCCs e centróide espectral do áudio. Normaliza as features.
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
    """Aplica augmentations no áudio."""
    try:
        return augmentations(samples=data, sample_rate=sr)
    except Exception as e:
        logging.error(f"Erro ao aumentar áudio: {e}")
        return data

def gerar_espectrograma(data, sr):
    """Gera espectrograma e retorna como PIL Image."""
    try:
        S = librosa.stft(data, n_fft=1024, hop_length=512)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        plt.figure(figsize=(2,2), dpi=100)  # Ajuste o tamanho conforme necessário
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='gray')
        plt.axis('off')  # Remove eixos para deixar só a imagem
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
    """Visualiza diferentes representações do áudio."""
    try:
        # Forma de onda usando waveshow
        fig_wave, ax_wave = plt.subplots(figsize=(8,4))
        librosa.display.waveshow(data, sr=sr, ax=ax_wave)
        ax_wave.set_title("Forma de Onda no Tempo")
        ax_wave.set_xlabel("Tempo (s)")
        ax_wave.set_ylabel("Amplitude")
        st.pyplot(fig_wave)
        plt.close(fig_wave)

        # FFT (Espectro)
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

        # MFCCs Plot
        mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
        fig_mfcc, ax_mfcc = plt.subplots(figsize=(8,4))
        img_mfcc = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax_mfcc)
        ax_mfcc.set_title("MFCCs")
        fig_mfcc.colorbar(img_mfcc, ax=ax_mfcc)
        st.pyplot(fig_mfcc)
        plt.close(fig_mfcc)
    except Exception as e:
        st.error(f"Erro na visualização do áudio: {e}")
        logging.error(f"Erro na visualização do áudio: {e}")

def visualizar_exemplos_classe(df, y, classes, augmentation=False, sr=22050, metodo='CNN'):
    """
    Visualiza pelo menos um exemplo de cada classe original e, se augmentation=True,
    também um exemplo aumentado.
    """
    classes_indices = {c: np.where(y == i)[0] for i, c in enumerate(classes)}

    st.markdown("### Visualizações Espectrais e MFCCs de Exemplos do Dataset (1 de cada classe original e 1 de cada classe aumentada)")

    if metodo in ['CNN', 'ResNet18', 'ResNet-18']:
        transforms_aug = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0),
            PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
            Shift(min_shift=-0.5, max_shift=0.5, p=1.0),
        ])
    else:
        transforms_aug = None

    for c in classes:
        st.markdown(f"#### Classe: {c}")
        indices_classe = classes_indices[c]
        st.write(f"Número de amostras para a classe '{c}': {len(indices_classe)}")
        if len(indices_classe) == 0:
            st.warning(f"**Nenhum exemplo encontrado para a classe {c}.**")
            continue
        # Seleciona um exemplo aleatório
        idx_original = random.choice(indices_classe)
        st.write(f"Selecionando índice original: {idx_original}")
        if idx_original >= len(df):
            st.error(f"Índice {idx_original} está fora do intervalo do DataFrame.")
            logging.error(f"Índice {idx_original} está fora do intervalo do DataFrame.")
            continue
        arquivo_original = df.iloc[idx_original]['caminho_arquivo']
        data_original, sr_original = carregar_audio(arquivo_original, sr=None)
        if data_original is not None and sr_original is not None:
            st.markdown(f"**Exemplo Original:** {os.path.basename(arquivo_original)}")
            visualizar_audio(data_original, sr_original)
            if metodo in ['ResNet18', 'ResNet-18']:
                # Gerar e exibir espectrograma
                espectrograma = gerar_espectrograma(data_original, sr_original)
                if espectrograma:
                    st.image(espectrograma, caption="Espectrograma Original", use_column_width=True)
        else:
            st.warning(f"**Não foi possível carregar o áudio da classe {c}.**")

        if augmentation and transforms_aug is not None:
            try:
                # Seleciona outro exemplo aleatório para augmentation
                idx_aug = random.choice(indices_classe)
                st.write(f"Selecionando índice para augmentation: {idx_aug}")
                if idx_aug >= len(df):
                    st.error(f"Índice {idx_aug} está fora do intervalo do DataFrame.")
                    logging.error(f"Índice {idx_aug} está fora do intervalo do DataFrame.")
                    continue
                arquivo_aug = df.iloc[idx_aug]['caminho_arquivo']
                data_aug, sr_aug = carregar_audio(arquivo_aug, sr=None)
                if data_aug is not None and sr_aug is not None:
                    aug_data = aumentar_audio(data_aug, sr_aug, transforms_aug)
                    st.markdown(f"**Exemplo Aumentado a partir de:** {os.path.basename(arquivo_aug)}")
                    visualizar_audio(aug_data, sr_aug)
                    if metodo in ['ResNet18', 'ResNet-18']:
                        # Gerar e exibir espectrograma aumentado
                        espectrograma_aug = gerar_espectrograma(aug_data, sr_aug)
                        if espectrograma_aug:
                            st.image(espectrograma_aug, caption="Espectrograma Aumentado", use_column_width=True)
                else:
                    st.warning(f"**Não foi possível carregar o áudio para augmentation da classe {c}.**")
            except Exception as e:
                st.warning(f"**Erro ao aplicar augmentation na classe {c}: {e}**")
                logging.error(f"Erro ao aplicar augmentation na classe {c}: {e}")

def escolher_k_kmeans(X_original, max_k=10):
    """
    Escolher k para K-Means automaticamente de acordo com o dataset.
    Usaremos o coeficiente de silhueta para determinar o melhor k entre 2 e max_k.
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

# Dataset personalizado para ResNet-18
class AudioSpectrogramDataset(Dataset):
    def __init__(self, df, classes, transform=None):
        self.df = df
        self.classes = classes
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(classes)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        arquivo = self.df.iloc[idx]['caminho_arquivo']
        classe = self.df.iloc[idx]['classe']
        data, sr = carregar_audio(arquivo, sr=None)
        if data is None or sr is None:
            return None, -1
        espectrograma = gerar_espectrograma(data, sr)
        if espectrograma is None:
            return None, -1
        if self.transform:
            espectrograma = self.transform(espectrograma)
        label = self.label_encoder.transform([classe])[0]
        return espectrograma, label

def classificar_audio(SEED):
    with st.expander("Classificação de Novo Áudio com Modelo Treinado"):
        st.markdown("### Instruções para Classificar Áudio")
        st.markdown("""
        **Passo 1:** Upload do modelo treinado (.keras, .h5, ou .pth) e classes (classes.txt).  
        **Passo 2:** Escolha o método de classificação (CNN personalizada ou ResNet-18).  
        **Passo 3:** Upload do áudio a ser classificado.  
        **Passo 4:** O app extrai features ou gera espectrogramas e prediz a classe.
        """)

        metodo_classificacao = st.selectbox(
            "Escolha o Método de Classificação:",
            ["CNN Personalizada", "ResNet-18"],
            help="Selecione entre a CNN personalizada ou a ResNet-18 pré-treinada para classificação."
        )

        modelo_file = st.file_uploader("Upload do Modelo (.keras, .h5, .pth)", type=["keras","h5","pth"])
        classes_file = st.file_uploader("Upload do Arquivo de Classes (classes.txt)", type=["txt"])

        if modelo_file is not None and classes_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(modelo_file.name)[1]) as tmp_model:
                tmp_model.write(modelo_file.read())
                caminho_modelo = tmp_model.name

            try:
                if metodo_classificacao == "CNN Personalizada":
                    modelo = tf.keras.models.load_model(caminho_modelo, compile=False)
                elif metodo_classificacao == "ResNet-18":
                    # Carregar modelo ResNet-18 pré-treinado e ajustar
                    num_classes_placeholder = 1  # Placeholder, será ajustado após carregar as classes
                    modelo = models.resnet18(pretrained=False)
                    # A última camada será ajustada após carregar as classes
                logging.info("Modelo carregado com sucesso.")
                st.success("Modelo carregado com sucesso!")
                # Verificar a saída do modelo
                if metodo_classificacao == "CNN Personalizada":
                    st.write(f"Modelo carregado com saída: {modelo.output_shape}")
                elif metodo_classificacao == "ResNet-18":
                    st.write("Modelo ResNet-18 carregado.")
            except Exception as e:
                st.error(f"Erro ao carregar o modelo: {e}")
                logging.error(f"Erro ao carregar o modelo: {e}")
                return

            try:
                classes = classes_file.read().decode("utf-8").splitlines()
                if not classes:
                    st.error("O arquivo de classes está vazio.")
                    return
                logging.info("Arquivo de classes carregado com sucesso.")
            except Exception as e:
                st.error(f"Erro ao ler o arquivo de classes: {e}")
                logging.error(f"Erro ao ler o arquivo de classes: {e}")
                return

            # Verificar se o número de classes corresponde ao número de saídas do modelo
            num_classes_file = len(classes)
            if metodo_classificacao == "CNN Personalizada":
                try:
                    num_classes_model = modelo.output_shape[-1]
                    st.write(f"Número de classes no arquivo: {num_classes_file}")
                    st.write(f"Número de saídas no modelo: {num_classes_model}")
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
                    # Ajustar a última camada para o número de classes
                    modelo.fc = torch.nn.Linear(modelo.fc.in_features, num_classes_file)
                    modelo = modelo.to('cuda' if torch.cuda.is_available() else 'cpu')
                    logging.info("Última camada da ResNet-18 ajustada para o número de classes.")
                    st.success("Última camada da ResNet-18 ajustada para o número de classes.")
                except Exception as e:
                    st.error(f"Erro ao ajustar a ResNet-18: {e}")
                    logging.error(f"Erro ao ajustar a ResNet-18: {e}")
                    return

            st.markdown("**Modelo e Classes Carregados!**")
            st.markdown(f"**Classes:** {', '.join(classes)}")

            audio_file = st.file_uploader("Upload do Áudio para Classificação", type=["wav","mp3","flac","ogg","m4a"])
            if audio_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_audio:
                    tmp_audio.write(audio_file.read())
                    caminho_audio = tmp_audio.name

                data, sr = carregar_audio(caminho_audio, sr=None)
                if data is not None:
                    if metodo_classificacao == "CNN Personalizada":
                        ftrs = extrair_features(data, sr, use_mfcc=True, use_spectral_centroid=True)
                        if ftrs is not None:
                            ftrs = ftrs.reshape(1, -1, 1)
                            try:
                                pred = modelo.predict(ftrs)
                                pred_class = np.argmax(pred, axis=1)
                                pred_label = classes[pred_class[0]]
                                confidence = pred[0][pred_class[0]] * 100
                                st.markdown(f"**Classe Predita:** {pred_label} (Confiança: {confidence:.2f}%)")

                                # Gráfico de Probabilidades
                                fig_prob, ax_prob = plt.subplots(figsize=(8,4))
                                ax_prob.bar(classes, pred[0], color='skyblue')
                                ax_prob.set_title("Probabilidades por Classe")
                                ax_prob.set_ylabel("Probabilidade")
                                plt.xticks(rotation=45)
                                st.pyplot(fig_prob)
                                plt.close(fig_prob)

                                # Reprodução e Visualização do Áudio
                                st.audio(caminho_audio)
                                visualizar_audio(data, sr)
                            except Exception as e:
                                st.error(f"Erro na predição: {e}")
                                logging.error(f"Erro na predição: {e}")
                        else:
                            st.error("Não foi possível extrair features do áudio.")
                            logging.warning("Não foi possível extrair features do áudio.")
                    elif metodo_classificacao == "ResNet-18":
                        try:
                            # Gerar espectrograma
                            espectrograma = gerar_espectrograma(data, sr)
                            if espectrograma:
                                # Transformações para ResNet-18
                                transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                ])
                                espectrograma = transform(espectrograma).unsqueeze(0)  # Adiciona batch dimension

                                # Carregar o modelo (já ajustado anteriormente)
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

                                    # Gráfico de Probabilidades
                                    fig_prob, ax_prob = plt.subplots(figsize=(8,4))
                                    ax_prob.bar(classes, probs.cpu().numpy()[0], color='skyblue')
                                    ax_prob.set_title("Probabilidades por Classe")
                                    ax_prob.set_ylabel("Probabilidade")
                                    plt.xticks(rotation=45)
                                    st.pyplot(fig_prob)
                                    plt.close(fig_prob)

                                    # Reprodução e Visualização do Áudio
                                    st.audio(caminho_audio)
                                    visualizar_audio(data, sr)

                                    # Exibir espectrograma
                                    # Converter tensor para imagem para exibição
                                    espectrograma_img = espectrograma.cpu().squeeze().permute(1,2,0).numpy()
                                    # Desnormalizar para visualização
                                    espectrograma_img = np.clip(espectrograma_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
                                    st.image(espectrograma_img, caption="Espectrograma Classificado", use_column_width=True)
                        except Exception as e:
                            st.error(f"Erro na predição ResNet-18: {e}")
                            logging.error(f"Erro na predição ResNet-18: {e}")
                else:
                    st.error("Não foi possível carregar o áudio.")
                    logging.warning("Não foi possível carregar o áudio.")

def treinar_modelo(SEED):
    with st.expander("Treinamento do Modelo CNN ou ResNet-18"):
        st.markdown("### Instruções Passo a Passo")
        st.markdown("""
        **Passo 1:** Upload do dataset .zip (pastas=classes).  
        **Passo 2:** Escolha o método de treinamento (CNN personalizada ou ResNet-18).  
        **Passo 3:** Ajuste os parâmetros no sidebar.  
        **Passo 4:** Clique em 'Treinar Modelo'.  
        **Passo 5:** Analise métricas, matriz de confusão, histórico, SHAP e clustering.
        """)

        metodo_treinamento = st.selectbox(
            "Escolha o Método de Treinamento:",
            ["CNN Personalizada", "ResNet-18"],
            help="Selecione entre treinar uma CNN personalizada ou utilizar a ResNet-18 pré-treinada."
        )

        # Checkbox para permitir parar o treinamento
        stop_training_choice = st.sidebar.checkbox("Permitir Parar Treinamento a Qualquer Momento", value=False)

        if stop_training_choice:
            st.sidebar.write("Durante o treinamento, caso deseje parar, clique no botão abaixo:")
            stop_button = st.sidebar.button("Parar Treinamento Agora")
            if stop_button:
                st.session_state.stop_training = True
        else:
            st.session_state.stop_training = False

        st.markdown("### Passo 1: Upload do Dataset (ZIP)")
        zip_upload = st.file_uploader("Upload do ZIP", type=["zip"])

        if zip_upload is not None:
            try:
                st.write("Extraindo o Dataset...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                    tmp_zip.write(zip_upload.read())
                    caminho_zip = tmp_zip.name

                diretorio_extracao = tempfile.mkdtemp()
                with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
                    zip_ref.extractall(diretorio_extracao)
                caminho_base = diretorio_extracao

                categorias = [d for d in os.listdir(caminho_base) if os.path.isdir(os.path.join(caminho_base, d))]
                if len(categorias) == 0:
                    st.error("Nenhuma subpasta encontrada no ZIP.")
                    return

                st.success("Dataset extraído!")
                st.write(f"Classes encontradas: {', '.join(categorias)}")

                caminhos_arquivos = []
                labels = []
                for cat in categorias:
                    caminho_cat = os.path.join(caminho_base, cat)
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

                if metodo_treinamento == "CNN Personalizada":
                    st.write("Extraindo Features (MFCCs, Centróide)...")
                    X = []
                    y_valid = []
                    for i, row in df.iterrows():
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
                                # Salvar espectrograma temporariamente
                                temp_dir = tempfile.mkdtemp()
                                img_path = os.path.join(temp_dir, f"class_{y[i]}_img_{i}.png")
                                espectrograma.save(img_path)
                                X.append(img_path)
                                y_valid.append(y[i])

                st.write(f"Dados Processados: {len(X)} amostras.")

                st.sidebar.markdown("**Configurações de Treinamento:**")

                # Parâmetros de Treinamento
                num_epochs = st.sidebar.slider("Número de Épocas:", 10, 500, 50, 10)
                batch_size = st.sidebar.selectbox("Batch:", [8,16,32,64,128],0)

                treino_percentage = st.sidebar.slider("Treino (%)",50,90,70,5)
                valid_percentage = st.sidebar.slider("Validação (%)",5,30,15,5)
                test_percentage = 100 - (treino_percentage + valid_percentage)
                if test_percentage < 0:
                    st.sidebar.error("Treino + Validação > 100%")
                    st.stop()
                st.sidebar.write(f"Teste (%)={test_percentage}%")

                augment_factor = st.sidebar.slider("Fator Aumento:",1,100,10,1)
                dropout_rate = st.sidebar.slider("Dropout:",0.0,0.9,0.4,0.05)

                regularization_type = st.sidebar.selectbox("Regularização:",["None","L1","L2","L1_L2"],0)
                if regularization_type == "L1":
                    l1_regularization = st.sidebar.slider("L1:",0.0,0.1,0.001,0.001)
                    l2_regularization = 0.0
                elif regularization_type == "L2":
                    l2_regularization = st.sidebar.slider("L2:",0.0,0.1,0.001,0.001)
                    l1_regularization = 0.0
                elif regularization_type == "L1_L2":
                    l1_regularization = st.sidebar.slider("L1:",0.0,0.1,0.001,0.001)
                    l2_regularization = st.sidebar.slider("L2:",0.0,0.1,0.001,0.001)
                else:
                    l1_regularization = 0.0
                    l2_regularization = 0.0

                # Opções de Fine-Tuning Adicionais
                st.sidebar.markdown("**Fine-Tuning Adicional:**")
                learning_rate = st.sidebar.slider("Taxa de Aprendizado:", 1e-5, 1e-2, 1e-3, step=1e-5, format="%.5f")
                optimizer_choice = st.sidebar.selectbox("Otimização:", ["Adam", "SGD", "RMSprop"],0)

                enable_augmentation = st.sidebar.checkbox("Data Augmentation", True)
                if enable_augmentation:
                    adicionar_ruido = st.sidebar.checkbox("Ruído Gaussiano", True)
                    estiramento_tempo = st.sidebar.checkbox("Time Stretch", True)
                    alteracao_pitch = st.sidebar.checkbox("Pitch Shift", True)
                    deslocamento = st.sidebar.checkbox("Deslocamento", True)

                cross_validation = st.sidebar.checkbox("k-Fold?", False)
                if cross_validation:
                    k_folds = st.sidebar.number_input("Folds:",2,10,5,1)
                else:
                    k_folds = 1

                balance_classes = st.sidebar.selectbox("Balanceamento:",["Balanced","None"],0)

                if enable_augmentation and metodo_treinamento == "CNN Personalizada":
                    st.write("Aumentando Dados...")
                    X_aug = []
                    y_aug = []
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

                if metodo_treinamento == "CNN Personalizada" and enable_augmentation and X_aug and y_aug:
                    X_aug = np.array(X_aug)
                    y_aug = np.array(y_aug)
                    st.write(f"Dados Aumentados: {X_aug.shape}")
                    X_combined = np.concatenate((X, X_aug), axis=0)
                    y_combined = np.concatenate((y_valid, y_aug), axis=0)
                elif metodo_treinamento == "ResNet-18" and enable_augmentation and X_aug and y_aug:
                    # Aumento de dados para ResNet-18 (espectrogramas)
                    # Para simplificar, vamos aumentar replicando espectrogramas com augmentations
                    # Isso pode ser otimizado conforme necessidade
                    st.write("Aumento de dados para ResNet-18 não implementado.")
                    X_combined = X
                    y_combined = y_valid
                else:
                    X_combined = X
                    y_combined = y_valid

                st.write("Dividindo Dados...")
                if metodo_treinamento == "CNN Personalizada":
                    X_train, X_temp, y_train, y_temp = train_test_split(
                        X_combined, y_combined, 
                        test_size=(100 - treino_percentage)/100.0,
                        random_state=SEED, 
                        stratify=y_combined
                    )
                elif metodo_treinamento == "ResNet-18":
                    # Para ResNet-18, X_combined contém caminhos de imagens
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

                # Para ResNet-18, criamos datasets personalizados
                if metodo_treinamento == "ResNet-18":
                    # Definir transformações
                    transform_train = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                    transform_val = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])

                    # Criar Datasets
                    dataset_train = AudioSpectrogramDataset(pd.DataFrame({'caminho_arquivo': X_train, 'classe': y_train}), classes, transform=transform_train)
                    dataset_val = AudioSpectrogramDataset(pd.DataFrame({'caminho_arquivo': X_val, 'classe': y_val}), classes, transform=transform_val)
                    dataset_test = AudioSpectrogramDataset(pd.DataFrame({'caminho_arquivo': X_test, 'classe': y_test}), classes, transform=transform_val)

                    # Criar DataLoaders
                    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
                    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=2)
                    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)

                else:
                    # Para CNN personalizada, preparar dados
                    X_train_final = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                    X_val_final = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
                    X_test_final = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

                st.sidebar.markdown("**Configurações de Treinamento Adicionais:**")

                # Opções de Fine-Tuning Adicionais
                st.sidebar.markdown("**Fine-Tuning Adicional:**")
                learning_rate = st.sidebar.slider("Taxa de Aprendizado:", 1e-5, 1e-2, 1e-3, step=1e-5, format="%.5f")
                optimizer_choice = st.sidebar.selectbox("Otimização:", ["Adam", "SGD", "RMSprop"],0)

                cross_validation = st.sidebar.checkbox("k-Fold?", False)
                if cross_validation:
                    k_folds = st.sidebar.number_input("Folds:",2,10,5,1)
                else:
                    k_folds = 1

                balance_classes = st.sidebar.selectbox("Balanceamento:",["Balanced","None"],0)

                # Balanceamento de classes
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
                    # Opções de Augmentation já aplicadas anteriormente
                    pass

                if metodo_treinamento == "CNN Personalizada":
                    # Definir a arquitetura da CNN personalizada
                    num_conv_layers = st.sidebar.slider("Conv Layers",1,5,2,1)
                    conv_filters_str = st.sidebar.text_input("Filtros (vírgula):","64,128")
                    conv_kernel_size_str = st.sidebar.text_input("Kernel (vírgula):","10,10")
                    conv_filters = [int(f.strip()) for f in conv_filters_str.split(',')]
                    conv_kernel_size = [int(k.strip()) for k in conv_kernel_size_str.split(',')]

                    # Ajusta o tamanho do kernel se necessário
                    input_length = X_train_final.shape[1]
                    for i in range(num_conv_layers):
                        if conv_kernel_size[i] > input_length:
                            conv_kernel_size[i] = input_length

                    num_dense_layers = st.sidebar.slider("Dense Layers:",1,3,1,1)
                    dense_units_str = st.sidebar.text_input("Neurônios Dense (vírgula):","64")
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

                    # Configuração do Otimizador com Taxa de Aprendizado
                    if optimizer_choice == "Adam":
                        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                    elif optimizer_choice == "SGD":
                        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
                    elif optimizer_choice == "RMSprop":
                        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
                    else:
                        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # Default

                    modelo_cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

                elif metodo_treinamento == "ResNet-18":
                    # Configurar transformações para treinamento e validação
                    transform_train = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                    transform_val = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])

                    # Atualizar Datasets com transformações
                    dataset_train = AudioSpectrogramDataset(pd.DataFrame({'caminho_arquivo': X_train, 'classe': y_train}), classes, transform=transform_train)
                    dataset_val = AudioSpectrogramDataset(pd.DataFrame({'caminho_arquivo': X_val, 'classe': y_val}), classes, transform=transform_val)
                    dataset_test = AudioSpectrogramDataset(pd.DataFrame({'caminho_arquivo': X_test, 'classe': y_test}), classes, transform=transform_val)

                    # Criar DataLoaders
                    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
                    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=2)
                    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)

                    # Carregar ResNet-18 pré-treinada
                    modelo_resnet = models.resnet18(pretrained=True)
                    modelo_resnet.fc = torch.nn.Linear(modelo_resnet.fc.in_features, len(classes))
                    modelo_resnet = modelo_resnet.to('cuda' if torch.cuda.is_available() else 'cpu')

                    # Definir otimizador
                    if optimizer_choice == "Adam":
                        optimizer = torch.optim.Adam(modelo_resnet.parameters(), lr=learning_rate)
                    elif optimizer_choice == "SGD":
                        optimizer = torch.optim.SGD(modelo_resnet.parameters(), lr=learning_rate, momentum=0.9)
                    elif optimizer_choice == "RMSprop":
                        optimizer = torch.optim.RMSprop(modelo_resnet.parameters(), lr=learning_rate)
                    else:
                        optimizer = torch.optim.Adam(modelo_resnet.parameters(), lr=learning_rate)  # Default

                    # Definir critério de perda
                    criterion = torch.nn.CrossEntropyLoss(weight=class_weight if balance_classes == "Balanced" else None)

                # Callbacks e outros
                diretorio_salvamento = 'modelos_salvos'
                if not os.path.exists(diretorio_salvamento):
                    os.makedirs(diretorio_salvamento)

                es_monitor = st.sidebar.selectbox("Monitor (Early Stopping):", ["val_loss","val_accuracy"],0)
                es_patience = st.sidebar.slider("Patience:",1,20,5,1)
                es_mode = st.sidebar.selectbox("Mode:",["min","max"],0)

                if metodo_treinamento == "CNN Personalizada":
                    checkpointer = ModelCheckpoint(
                        os.path.join(diretorio_salvamento,'modelo_agua_aumentado.keras'),
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

                elif metodo_treinamento == "ResNet-18":
                    # Para ResNet-18, não há callbacks padrão no PyTorch, mas podemos implementar Early Stopping manualmente
                    pass

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

                        # Salvando o modelo
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

                        # Salvando as classes
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

                                # Plot das curvas de treinamento
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
                                    st.write(f"Número de shap_values: {num_shap_values}, classes: {num_classes}")

                                    if num_shap_values == num_classes:
                                        for class_idx, class_name in enumerate(classes):
                                            st.write(f"**Classe: {class_name}**")
                                            fig_shap = plt.figure()
                                            shap.summary_plot(shap_values[class_idx], X_sample, show=False)
                                            st.pyplot(fig_shap)
                                            plt.close(fig_shap)
                                    elif num_shap_values == 1 and num_classes == 2:
                                        # Classificação binária, shap_values[0] corresponde à classe positiva
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

                                melhor_k = escolher_k_kmeans(X_combined, max_k=10)
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
                                st.error(f"Erro durante a avaliação do modelo CNN: {e}")
                                logging.error(f"Erro durante a avaliação do modelo CNN: {e}")

                    elif metodo_treinamento == "ResNet-18":
                        # Treinar modelo ResNet-18
                        try:
                            from torch.optim import lr_scheduler
                        except ImportError:
                            st.error("PyTorch não está instalado. Por favor, instale PyTorch para usar a ResNet-18.")
                            return

                        # Definir scheduler
                        scheduler_resnet = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=1)

                        best_acc = 0.0

                        for epoch in range(num_epochs):
                            if st.session_state.stop_training:
                                st.warning("Treinamento Parado pelo Usuário!")
                                break
                            st.write(f"Epoch {epoch+1}/{num_epochs}")
                            st.write('-' * 10)

                            # Treinamento
                            modelo_resnet.train()
                            running_loss = 0.0
                            running_corrects = 0

                            for inputs, labels in loader_train:
                                if inputs is None:
                                    continue
                                inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
                                labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

                                optimizer.zero_grad()

                                outputs = modelo_resnet(inputs)
                                _, preds = torch.max(outputs, 1)
                                loss = criterion(outputs, labels)

                                loss.backward()
                                optimizer.step()

                                running_loss += loss.item() * inputs.size(0)
                                running_corrects += torch.sum(preds == labels.data)

                            epoch_loss = running_loss / len(loader_train.dataset)
                            epoch_acc = running_corrects.double() / len(loader_train.dataset)

                            st.write(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                            # Validação
                            modelo_resnet.eval()
                            val_loss = 0.0
                            val_corrects = 0

                            with torch.no_grad():
                                for inputs, labels in loader_val:
                                    if inputs is None:
                                        continue
                                    inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
                                    labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

                                    outputs = modelo_resnet(inputs)
                                    _, preds = torch.max(outputs, 1)
                                    loss = criterion(outputs, labels)

                                    val_loss += loss.item() * inputs.size(0)
                                    val_corrects += torch.sum(preds == labels.data)

                            epoch_val_loss = val_loss / len(loader_val.dataset)
                            epoch_val_acc = val_corrects.double() / len(loader_val.dataset)

                            st.write(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')

                            # Scheduler step
                            scheduler_resnet.step(epoch_val_loss)

                            # Checkpoint
                            if epoch_val_acc > best_acc:
                                best_acc = epoch_val_acc
                                torch.save(modelo_resnet.state_dict(), os.path.join(diretorio_salvamento, 'best_resnet18.pth'))
                                st.write(f"Melhor acurácia atualizada: {best_acc:.4f}")

                        if not st.session_state.stop_training:
                            st.success("Treino concluído!")

                            # Carregar o melhor modelo
                            try:
                                modelo_resnet.load_state_dict(torch.load(os.path.join(diretorio_salvamento, 'best_resnet18.pth')))
                                modelo_resnet.eval()
                            except Exception as e:
                                st.error(f"Erro ao carregar o melhor modelo: {e}")
                                logging.error(f"Erro ao carregar o melhor modelo: {e}")
                                return

                            # Salvando o modelo
                            try:
                                buffer = io.BytesIO()
                                torch.save(modelo_resnet.state_dict(), buffer)
                                buffer.seek(0)
                                st.download_button("Download Modelo ResNet-18 (.pth)", data=buffer, file_name="best_resnet18.pth")
                            except Exception as e:
                                st.error(f"Erro ao salvar o modelo ResNet-18: {e}")
                                logging.error(f"Erro ao salvar o modelo ResNet-18: {e}")

                            # Salvando as classes
                            try:
                                classes_str = "\n".join(classes)
                                st.download_button("Download Classes (classes.txt)", data=classes_str, file_name="classes.txt")
                            except Exception as e:
                                st.error(f"Erro ao salvar o arquivo de classes: {e}")
                                logging.error(f"Erro ao salvar o arquivo de classes: {e}")

                            # Avaliação no conjunto de teste
                            try:
                                st.markdown("### Avaliação do Modelo")
                                running_loss = 0.0
                                running_corrects = 0

                                for inputs, labels in loader_test:
                                    if inputs is None:
                                        continue
                                    inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
                                    labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

                                    outputs = modelo_resnet(inputs)
                                    _, preds = torch.max(outputs, 1)
                                    loss = criterion(outputs, labels)

                                    running_loss += loss.item() * inputs.size(0)
                                    running_corrects += torch.sum(preds == labels.data)

                                test_loss = running_loss / len(loader_test.dataset)
                                test_acc = running_corrects.double() / len(loader_test.dataset)

                                st.write(f"Acurácia Teste: {test_acc:.4f}")

                                # Previsões para métricas adicionais
                                y_pred = []
                                y_true = []

                                for inputs, labels in loader_test:
                                    if inputs is None:
                                        continue
                                    inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
                                    labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

                                    outputs = modelo_resnet(inputs)
                                    _, preds = torch.max(outputs, 1)

                                    y_pred.extend(preds.cpu().numpy())
                                    y_true.extend(labels.cpu().numpy())

                                f1_val = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                                prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                                rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)

                                st.write(f"F1-score: {f1_val*100:.2f}%")
                                st.write(f"Precisão: {prec*100:.2f}%")
                                st.write(f"Recall: {rec*100:.2f}%")

                                st.markdown("### Matriz de Confusão")
                                cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
                                cm_df = pd.DataFrame(cm, index=classes, columns=classes)
                                fig_cm, ax_cm = plt.subplots(figsize=(12,8))
                                sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                ax_cm.set_title("Matriz de Confusão", fontsize=16)
                                ax_cm.set_xlabel("Classe Prevista", fontsize=14)
                                ax_cm.set_ylabel("Classe Real", fontsize=14)
                                st.pyplot(fig_cm)
                                plt.close(fig_cm)

                                st.markdown("### Histórico de Treinamento")
                                # Para ResNet-18, histórico não está disponível diretamente
                                st.warning("Histórico de Treinamento não disponível para ResNet-18.")

                                st.markdown("### Explicabilidade com SHAP")
                                st.write("SHAP não está implementado para ResNet-18 neste exemplo.")

                                st.markdown("### Análise de Clusters (K-Means e Hierárquico)")
                                st.write("""
                                Clustering revela como dados se agrupam.  
                                Determinaremos k automaticamente usando o coeficiente de silhueta.
                                """)

                                melhor_k = escolher_k_kmeans(X_combined, max_k=10)
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

                                st.markdown("### Visualização de Exemplos")
                                visualizar_exemplos_classe(df, y_valid, classes, augmentation=enable_augmentation, sr=22050, metodo=metodo_treinamento)

                            except Exception as e:
                                st.error(f"Erro durante a avaliação do modelo ResNet-18: {e}")
                                logging.error(f"Erro durante a avaliação do modelo ResNet-18: {e}")

            except Exception as e:
                st.error(f"Erro: {e}")
                logging.error(f"Erro: {e}")

            # Limpeza de memória e arquivos temporários
            try:
                gc.collect()
                os.remove(caminho_zip)
                for cat in categorias:
                    caminho_cat = os.path.join(caminho_base, cat)
                    for arquivo in os.listdir(caminho_cat):
                        os.remove(os.path.join(caminho_cat, arquivo))
                    os.rmdir(caminho_cat)
                os.rmdir(caminho_base)
                logging.info("Processo concluído.")
            except Exception as e:
                logging.error(f"Erro ao limpar arquivos temporários: {e}")

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

st.sidebar.header("Configurações Gerais")
with st.sidebar.expander("Parâmetro SEED e Reprodutibilidade"):
    st.markdown("**SEED** garante resultados reproduzíveis.")

seed_selection = st.sidebar.selectbox(
    "Escolha o valor do SEED:",
    options=seed_options,
    index=seed_options.index(default_seed),
    help="Define a semente para reprodutibilidade."
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

app_mode = st.sidebar.radio("Escolha a seção", ["Classificar Áudio", "Treinar Modelo"])

if app_mode == "Classificar Áudio":
    classificar_audio(SEED)
elif app_mode == "Treinar Modelo":
    treinar_modelo(SEED)

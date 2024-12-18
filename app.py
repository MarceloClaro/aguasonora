# Importação das bibliotecas necessárias
import random  # Para gerar números aleatórios e garantir reprodutibilidade
import numpy as np  # Para manipulação de arrays e operações matemáticas
import pandas as pd  # Para trabalhar com DataFrames e manipulação de dados
import matplotlib.pyplot as plt  # Para visualização de gráficos
import seaborn as sns  # Para visualização mais avançada de gráficos
import librosa  # Biblioteca para análise de áudio
import librosa.display  # Para exibição de espectrogramas e outros gráficos de áudio
import tensorflow as tf  # Framework para criar e treinar redes neurais (deep learning)
from tensorflow.keras.utils import to_categorical  # Para codificação de rótulos (one-hot encoding)
from sklearn.preprocessing import LabelEncoder  # Para codificar rótulos de classe
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score  # Para avaliação de desempenho do modelo
from sklearn.model_selection import train_test_split  # Para dividir os dados em treino e teste
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift  # Para aumentação de dados de áudio
import streamlit as st  # Biblioteca para criar a interface interativa
import tempfile  # Para trabalhar com arquivos temporários
from PIL import Image  # Biblioteca para manipulação de imagens (usada para espectrogramas)
import os  # Para operações com arquivos e diretórios
import logging  # Para registrar logs de execução

# ============================ Configurações Iniciais ============================

# Configuração de logging
logging.basicConfig(
    filename='experiment_logs.log',  # Arquivo de log onde os erros e eventos serão registrados
    filemode='a',  # Modo de abertura do arquivo de log (adicionando novas entradas)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Formato das mensagens no log
    level=logging.INFO  # Nível de log (INFO é o nível padrão para registrar mensagens informativas)
)

def set_seeds(seed: int):
    """
    Define as sementes para reprodutibilidade.

    Args:
        seed (int): Valor da semente.
    """
    random.seed(seed)  # Define a semente para números aleatórios em Python
    np.random.seed(seed)  # Define a semente para a geração de números aleatórios do NumPy
    tf.random.set_seed(seed)  # Define a semente para TensorFlow
    torch.manual_seed(seed)  # Define a semente para PyTorch
    if torch.cuda.is_available():  # Se houver GPU disponível
        torch.cuda.manual_seed_all(seed)  # Define a semente para todas as GPUs disponíveis
# ============================ Funções Auxiliares ============================

def carregar_audio(caminho_arquivo: str, sr: int = None):
    """
    Carrega um arquivo de áudio usando Librosa com logs detalhados.

    Args:
        caminho_arquivo (str): Caminho para o arquivo de áudio.
        sr (int, optional): Taxa de amostragem. Se None, usa a taxa original.

    Returns:
        tuple: Dados de áudio e taxa de amostragem, ou (None, None) em caso de erro.
    """
    if not os.path.exists(caminho_arquivo):
        error_msg = f"Arquivo de áudio não encontrado: {caminho_arquivo}"
        logging.error(error_msg)  # Registra o erro no log
        st.error(error_msg)  # Exibe o erro no Streamlit
        return None, None
    try:
        data, sr = librosa.load(caminho_arquivo, sr=sr, res_type='kaiser_fast', backend='soundfile')  # Carrega o áudio
        if len(data) == 0:
            error_msg = f"Arquivo de áudio vazio: {caminho_arquivo}"
            logging.error(error_msg)
            st.error(error_msg)
            return None, None
        logging.info(f"Áudio carregado com sucesso: {caminho_arquivo} (Duração: {len(data)/sr:.2f}s)")
        return data, sr  # Retorna os dados de áudio e a taxa de amostragem
    except Exception as e:
        error_msg = f"Erro ao carregar áudio {caminho_arquivo}: {e}"
        logging.error(error_msg)
        st.error(error_msg)
        return None, None


def extrair_features(data: np.ndarray, sr: int, use_mfcc: bool = True, use_spectral_centroid: bool = True):
    """
    Extrai features do áudio.

    Args:
        data (np.ndarray): Dados de áudio.
        sr (int): Taxa de amostragem.
        use_mfcc (bool): Se True, extrai MFCCs.
        use_spectral_centroid (bool): Se True, extrai centróide espectral.

    Returns:
        np.ndarray: Vetor de features normalizado, ou None em caso de erro.
    """
    try:
        features_list = []
        if use_mfcc:
            mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            features_list.append(mfccs_scaled)  # Extrai os MFCCs (coeficientes cepstrais)
        if use_spectral_centroid:
            centroid = librosa.feature.spectral_centroid(y=data, sr=sr)
            centroid_mean = np.mean(centroid, axis=1)
            features_list.append(centroid_mean)  # Extrai o centróide espectral
        if len(features_list) > 1:
            features_vector = np.concatenate(features_list, axis=0)
        else:
            features_vector = features_list[0]
        # Normalização das features
        features_vector = (features_vector - np.mean(features_vector)) / (np.std(features_vector) + 1e-9)
        return features_vector
    except Exception as e:
        logging.error(f"Erro ao extrair features: {e}")
        st.error(f"Erro ao extrair features: {e}")
        return None


def aumentar_audio(data: np.ndarray, sr: int, augmentations: Compose):
    """
    Aplica aumentação de dados no áudio.

    Args:
        data (np.ndarray): Dados de áudio.
        sr (int): Taxa de amostragem.
        augmentations (Compose): Pipeline de aumentação.

    Returns:
        np.ndarray: Áudio aumentado, ou o áudio original em caso de erro.
    """
    try:
        return augmentations(samples=data, sample_rate=sr)  # Aplica as aumentações no áudio
    except Exception as e:
        logging.error(f"Erro ao aumentar áudio: {e}")
        st.warning(f"Erro ao aumentar áudio: {e}")
        return data
#=================================================================================================
def gerar_espectrograma(data: np.ndarray, sr: int):
    """
    Gera um espectrograma a partir do áudio.

    Args:
        data (np.ndarray): Dados de áudio.
        sr (int): Taxa de amostragem.

    Returns:
        PIL.Image.Image: Imagem do espectrograma, ou None em caso de erro.
    """
    try:
        if len(data) == 0:
            logging.warning("Dados de áudio vazios.")
            st.warning("Dados de áudio vazios.")
            return None

        S = librosa.stft(data, n_fft=1024, hop_length=512)  # Transformada de Fourier de curto prazo
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)  # Converte a amplitude para dB

        # Plota o espectrograma
        plt.figure(figsize=(10, 4), dpi=100)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='gray')
        plt.axis('off')  # Remove os eixos
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        img = Image.open(buf).convert('RGB')  # Converte para uma imagem RGB
        logging.info("Espectrograma gerado com sucesso.")
        return img
    except Exception as e:
        error_msg = f"Erro ao gerar espectrograma: {e}"
        logging.error(error_msg)
        st.warning(error_msg)
        return None


def visualizar_audio(data: np.ndarray, sr: int):
    """
    Visualiza diferentes representações do áudio.

    Args:
        data (np.ndarray): Dados de áudio.
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
        error_msg = f"Erro ao visualizar áudio: {e}"
        logging.error(error_msg)
        st.warning(error_msg)
#============================================================================================
class AudioSpectrogramDataset(Dataset):
    """
    Dataset personalizado para espectrogramas de áudio.

    Args:
        df (pd.DataFrame): DataFrame com caminhos dos arquivos e classes.
        classes (list): Lista de classes.
        transform (callable, optional): Transformação a ser aplicada nas imagens.
    """
    def __init__(self, df: pd.DataFrame, classes: list, transform=None):
        self.df = df
        self.classes = classes
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(classes)
        self.indices_validos = self._filtrar_amostras_validas()

    def _filtrar_amostras_validas(self):
        """
        Filtra amostras válidas antes de criar o dataset.

        Returns:
            list: Índices das amostras válidas.
        """
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
            error_msg = f"Erro em __getitem__ para o índice {real_idx}: {e}"
            logging.error(error_msg)
            st.warning(error_msg)
            # Em vez de retornar None, levanta a exceção para que possa ser capturada no treinamento
            raise e
#=============================================================================================================
def escolher_k_kmeans(X_original: np.ndarray, y: np.ndarray, max_k: int = 10):
    """
    Determina o melhor número de clusters usando o coeficiente de silhueta.

    Args:
        X_original (np.ndarray): Features.
        y (np.ndarray): Labels.
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
#===================================================================================================
def verificar_contagem_classes(y: np.ndarray, k_folds: int = 5):
    """
    Verifica se todas as classes têm pelo menos k_folds amostras.

    Args:
        y (np.ndarray): Labels.
        k_folds (int, optional): Número de folds para cross-validation.

    Returns:
        bool: True se todas as classes têm pelo menos k_folds amostras, False caso contrário.
    """
    counts = np.bincount(y)
    if np.any(counts < k_folds):
        return False
    return True
#====================================================================================
def custom_collate(batch):
    """
    Função de collate personalizada para DataLoader.

    Args:
        batch (list): Lista de tuplas (imagem, label).

    Returns:
        tuple: Tensores de imagens e labels.
    """
    inputs, labels = zip(*batch)
    inputs = torch.stack(inputs, dim=0)
    labels = torch.tensor(labels)
    return inputs, labels
#=====================================================================================
def carregar_dados(zip_path: str):
    """
    Extrai e carrega dados do arquivo ZIP.

    Args:
        zip_path (str): Caminho para o arquivo ZIP.

    Returns:
        tuple: Diretório de extração e lista de categorias (pastas), ou (None, None) em caso de erro.
    """
    try:
        diretorio_extracao = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(diretorio_extracao)
        categorias = [d for d in os.listdir(diretorio_extracao) if os.path.isdir(os.path.join(diretorio_extracao, d))]
        logging.info(f"ZIP extraído para {diretorio_extracao} com categorias: {categorias}")
        return diretorio_extracao, categorias
    except Exception as e:
        error_msg = f"Erro ao extrair o ZIP: {e}"
        logging.error(error_msg)
        st.error(error_msg)
        return None, None
#=============================================================================================
def visualizar_exemplos_classe(df: pd.DataFrame, y_valid: np.ndarray, classes: list, augmentation: bool = False, sr: int = 22050, metodo: str = "CNN Personalizada"):
    """
    Visualiza exemplos de cada classe do dataset.

    Args:
        df (pd.DataFrame): DataFrame com caminhos dos arquivos e classes.
        y_valid (np.ndarray): Labels válidos.
        classes (list): Lista de classes.
        augmentation (bool, optional): Se True, mostra exemplos aumentados.
        sr (int, optional): Taxa de amostragem para visualização.
        metodo (str, optional): Método de treinamento utilizado.
    """
    try:
        st.markdown("### Exemplos por Classe")
        for cls in classes:
            st.subheader(f"Classe: {cls}")
            class_idx = np.where(classes == cls)[0][0]
            indices = np.where(y_valid == class_idx)[0]
            if len(indices) == 0:
                st.write("Nenhum exemplo disponível.")
                continue
            selecionados = np.random.choice(indices, min(5, len(indices)), replace=False)
            for idx in selecionados:
                arquivo = df.iloc[idx]['caminho_arquivo']
                data, sr = carregar_audio(arquivo, sr=sr)
                if data is not None:
                    st.audio(arquivo)
                    visualizar_audio(data, sr)
    except Exception as e:
        error_msg = f"Erro ao visualizar exemplos por classe: {e}"
        st.error(error_msg)
        logging.error(error_msg)
#==============================================================================================================
def classificar_audio(SEED: int):
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
                    modelo = models.resnet18(pretrained=False)
                    # A última camada será ajustada após carregar as classes
                logging.info("Modelo carregado com sucesso.")
                st.success("Modelo carregado com sucesso!")
            except Exception as e:
                error_msg = f"Erro ao carregar o modelo: {e}"
                st.error(error_msg)
                logging.error(error_msg)
                return

            try:
                classes = classes_file.read().decode("utf-8").splitlines()
                classes = [c.strip() for c in classes if c.strip()]
                if not classes:
                    st.error("O arquivo de classes está vazio.")
                    return
                logging.info("Arquivo de classes carregado com sucesso.")
            except Exception as e:
                error_msg = f"Erro ao ler o arquivo de classes: {e}"
                st.error(error_msg)
                logging.error(error_msg)
                return

            if metodo_classificacao == "CNN Personalizada":
                try:
                    num_classes_model = modelo.output_shape[-1]
                    num_classes_file = len(classes)
                    if num_classes_file != num_classes_model:
                        error_msg = f"Número de classes ({num_classes_file}) não corresponde ao número de saídas do modelo ({num_classes_model})."
                        st.error(error_msg)
                        logging.error(error_msg)
                        return
                except AttributeError as e:
                    error_msg = f"Erro ao verificar a saída do modelo: {e}"
                    st.error(error_msg)
                    logging.error(error_msg)
                    return
            elif metodo_classificacao == "ResNet-18":
                try:
                    num_classes = len(classes)
                    modelo.fc = torch.nn.Linear(modelo.fc.in_features, num_classes)
                    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
                    modelo = modelo.to(dispositivo)
                    logging.info("Última camada da ResNet-18 ajustada para o número de classes.")
                    st.success("Última camada da ResNet-18 ajustada para o número de classes.")
                except Exception as e:
                    error_msg = f"Erro ao ajustar a ResNet-18: {e}"
                    st.error(error_msg)
                    logging.error(error_msg)
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
                                        torch_transforms.RandomHorizontalFlip(),
                                        torch_transforms.ToTensor(),
                                        torch_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
                                    ])
                                    espectrograma = transform(espectrograma).unsqueeze(0)
                                    modelo.eval()
                                    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
                                    modelo = modelo.to(dispositivo)
                                    with torch.no_grad():
                                        espectrograma = espectrograma.to(dispositivo)
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

                                        # Convertendo o espectrograma para visualização
                                        # Como o modelo espera 3 canais (RGB), precisamos duplicar a imagem em 3 canais
                                        # Note que a transformação já converteu para tensor, então precisamos reverter
                                        espectrograma_img = espectrograma.cpu().squeeze().permute(1,2,0).numpy()
                                        # Revertendo a normalização
                                        espectrograma_img = np.clip(espectrograma_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
                                        st.image(espectrograma_img, caption="Espectrograma Classificado", use_column_width=True)
                            except Exception as e:
                                error_msg = f"Erro na predição ResNet-18: {e}"
                                st.error(error_msg)
                                logging.error(error_msg)
                    else:
                        st.error("Não foi possível carregar o áudio.")
                        logging.warning("Não foi possível carregar o áudio.")
                except Exception as e:
                    error_msg = f"Erro ao processar o áudio: {e}"
                    st.error(error_msg)
                    logging.error(error_msg)
#=========================================================================================================================================
def treinar_modelo(SEED: int):
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
                                logging.info(f"Features extraídas para: {arquivo}")
                            else:
                                logging.warning(f"Features não extraídas para: {arquivo}")
                        else:
                            logging.warning(f"Áudio não carregado para o arquivo: {arquivo}")
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
                                try:
                                    espectrograma.save(img_path)
                                    X.append(img_path)
                                    y_valid.append(y[i])
                                    logging.info(f"Espectrograma gerado para: {arquivo}, salvo em: {img_path}")
                                except Exception as e:
                                    error_msg = f"Erro ao salvar espectrograma para {arquivo}: {e}"
                                    logging.error(error_msg)
                                    st.warning(error_msg)
                            else:
                                logging.warning(f"Espectrograma não gerado para: {arquivo}")
                                st.warning(f"Espectrograma não gerado para: {arquivo}")
                        else:
                            logging.warning(f"Áudio não carregado para o arquivo: {arquivo}")
                            st.warning(f"Áudio não carregado para o arquivo: {arquivo}")
                    X = np.array(X)
                    y_valid = np.array(y_valid)

                st.write(f"Dados Processados: {len(X)} amostras.")

                if len(X) == 0:
                    st.error("Nenhuma amostra válida foi processada. Verifique os logs para mais detalhes.")
                    return

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
#============================================================================
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
                                        logging.info(f"Features aumentadas para: {arquivo}")
                                    else:
                                        logging.warning(f"Features aumentadas não extraídas para: {arquivo}")

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
                                try:
                                    espectrograma_aug.save(img_path)
                                    X_aug.append(img_path)
                                    y_aug.append(y[i])
                                    logging.info(f"Espectrograma aumentado gerado para: {arquivo}, salvo em: {img_path}")
                                except Exception as e:
                                    error_msg = f"Erro ao salvar espectrograma aumentado para {arquivo}: {e}"
                                    logging.error(error_msg)
                                    st.warning(error_msg)
                            else:
                                logging.warning(f"Espectrograma aumentado não gerado para: {arquivo}")
                                st.warning(f"Espectrograma aumentado não gerado para: {arquivo}")
                        else:
                            logging.warning(f"Áudio não carregado para o arquivo: {arquivo}")
                            st.warning(f"Áudio não carregado para o arquivo: {arquivo}")

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
                            st.warning(f"Arquivo de espectrograma inexistente: {img_path}")
                    X_combined = X_combined[valid_indices]
                    y_combined = y_combined[valid_indices]
                    st.write(f"Amostras válidas após verificação: {len(X_combined)}")

                # Verificar se todas as classes têm pelo menos k_folds amostras
                if not verificar_contagem_classes(y_combined, k_folds=k_folds):
                    st.error("Algumas classes têm menos amostras do que o número de folds.")
                    st.stop()

                st.write("Dividindo Dados...")
                if metodo_treinamento == "CNN Personalizada":
                    # Para CNN, os dados são features vetorizadas
                    X_train, X_temp, y_train, y_temp = train_test_split(
                        X_combined, y_combined, 
                        test_size=(100 - treino_percentage)/100.0,
                        random_state=SEED, 
                        stratify=y_combined
                    )
                elif metodo_treinamento == "ResNet-18":
                    # Para ResNet-18, os dados são caminhos de imagens
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
                    # Para CNN Personalizada, reshape os dados para (samples, features, 1)
                    X_train_final = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                    X_val_final = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
                    X_test_final = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
#===============================================================================================================
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
                    error_msg = f"Erro ao limpar arquivos temporários: {e}"
                    logging.error(error_msg)
                    st.warning(error_msg)



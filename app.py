import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display as ld
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import streamlit as st
import tempfile
from PIL import Image
import io
import torch
import zipfile
import gc

# Desativa CUDA para PyTorch se não necessário
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configurações para visualizações
sns.set(style='whitegrid', context='notebook')

# ==================== CONTROLE DE REPRODUTIBILIDADE ====================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==================== FUNÇÕES DE PROCESSAMENTO ====================

augment_default = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
])

def carregar_audio(caminho_arquivo, sr=None):
    """
    Carrega um arquivo de áudio.

    Parameters:
    - caminho_arquivo (str): Caminho para o arquivo de áudio.
    - sr (int, optional): Taxa de amostragem. Se None, usa a taxa original.

    Returns:
    - data (np.ndarray): Sinal de áudio.
    - sr (int): Taxa de amostragem.
    """
    try:
        data, sr = librosa.load(caminho_arquivo, sr=sr, res_type='kaiser_fast')
        return data, sr
    except Exception as e:
        st.error(f"Erro ao carregar o áudio {caminho_arquivo}: {e}")
        return None, None

def extrair_features(data, sr):
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

def aumentar_audio(data, sr, augmentations):
    """
    Aplica Data Augmentation ao sinal de áudio com as transformações selecionadas.

    Parameters:
    - data (np.ndarray): Sinal de áudio.
    - sr (int): Taxa de amostragem.
    - augmentations (Compose): Composição de transformações de aumento.

    Returns:
    - augmented_data (np.ndarray): Sinal de áudio aumentado.
    """
    try:
        augmented_data = augmentations(samples=data, sample_rate=sr)
        return augmented_data
    except Exception as e:
        st.error(f"Erro ao aplicar Data Augmentation: {e}")
        return data  # Retorna o original em caso de erro

def plot_forma_onda(data, sr, titulo="Forma de Onda"):
    """
    Plota a forma de onda do áudio.

    Parameters:
    - data (np.ndarray): Sinal de áudio.
    - sr (int): Taxa de amostragem.
    - titulo (str): Título do gráfico.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(titulo)
    ld.waveshow(data, sr=sr, ax=ax)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)
    plt.close(fig)

def plot_espectro_frequencias(data, sr, titulo="Espectro de Frequências"):
    """
    Plota o espectro de frequências do áudio.

    Parameters:
    - data (np.ndarray): Sinal de áudio.
    - sr (int): Taxa de amostragem.
    - titulo (str): Título do gráfico.
    """
    N = len(data)
    fft = np.fft.fft(data)
    fft = np.abs(fft[:N//2])  # Apenas a metade positiva do espectro
    freqs = np.fft.fftfreq(N, 1/sr)[:N//2]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(titulo)
    ax.plot(freqs, fft)
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

def plot_espectrograma(data, sr, titulo="Espectrograma (STFT)"):
    """
    Plota o espectrograma (STFT) do áudio.

    Parameters:
    - data (np.ndarray): Sinal de áudio.
    - sr (int): Taxa de amostragem.
    - titulo (str): Título do gráfico.
    """
    D = np.abs(librosa.stft(data))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(titulo)
    mappable = ld.specshow(DB, sr=sr, x_axis='time', y_axis='hz', cmap='magma', ax=ax)
    plt.colorbar(mappable=mappable, ax=ax, format='%+2.0f dB')
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Frequência (Hz)")
    st.pyplot(fig)
    plt.close(fig)

def plot_mfcc(data, sr, titulo="Espectrograma (MFCC)"):
    """
    Plota o espectrograma de MFCC do áudio.

    Parameters:
    - data (np.ndarray): Sinal de áudio.
    - sr (int): Taxa de amostragem.
    - titulo (str): Título do gráfico.
    """
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    mfccs_db = librosa.amplitude_to_db(np.abs(mfccs))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(titulo)
    mappable = ld.specshow(mfccs_db, x_axis='time', y_axis='mel', cmap='Spectral', sr=sr, ax=ax)
    plt.colorbar(mappable=mappable, ax=ax, format='%+2.f dB')
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Frequência (Mel)")
    st.pyplot(fig)
    plt.close(fig)

def plot_probabilidades_classes(class_probs, titulo="Probabilidades das Classes"):
    """
    Plota as probabilidades das classes em um gráfico de barras.

    Parameters:
    - class_probs (dict): Dicionário com as probabilidades de cada classe.
    - titulo (str): Título do gráfico.
    """
    classes = list(class_probs.keys())
    probs = list(class_probs.values())

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=classes, y=probs, hue=classes, palette='viridis', ax=ax, legend=False)
    ax.set_title(titulo)
    ax.set_xlabel("Classes")
    ax.set_ylabel("Probabilidade")
    ax.set_ylim(0, 1)  # Probabilidades entre 0 e 1
    for i, v in enumerate(probs):
        ax.text(i, v + 0.01, f"{v*100:.2f}%", ha='center', va='bottom')
    st.pyplot(fig)
    plt.close(fig)

def processar_novo_audio(caminho_audio, modelo, labelencoder):
    """
    Carrega, extrai features e classifica um novo arquivo de áudio.

    Parameters:
    - caminho_audio (str): Caminho para o arquivo de áudio.
    - modelo (tf.keras.Model ou torch.nn.Module): Modelo treinado para classificação.
    - labelencoder (LabelEncoder): Codificador de labels para decodificar classes.

    Returns:
    - pred_label (str): Rótulo da classe prevista.
    - confidence (float): Grau de confiança da previsão.
    - class_probs (dict): Dicionário com as probabilidades de cada classe.
    """
    # Carrega o áudio
    data, sr = carregar_audio(caminho_audio, sr=None)

    if data is None:
        return None, None, None

    # Extrai as features (MFCCs)
    mfccs = extrair_features(data, sr)

    if mfccs is None:
        return None, None, None

    # Ajusta o shape dos MFCCs para compatibilidade com o modelo
    # Conv1D espera dados com forma (samples, timesteps, features)
    # Aqui, timesteps correspondem ao número de features (MFCCs) e features=1
    mfccs = mfccs.reshape(1, -1, 1)  # Forma: (1, n_features, 1)

    # Realiza a predição usando o modelo treinado
    if isinstance(modelo, tf.keras.Model):
        prediction = modelo.predict(mfccs)
    elif isinstance(modelo, torch.nn.Module):
        modelo.eval()
        with torch.no_grad():
            mfccs_tensor = torch.tensor(mfccs, dtype=torch.float32)
            prediction = modelo(mfccs_tensor).numpy()
    else:
        st.error("Modelo não suportado.")
        return None, None, None

    # Obtém a classe com a maior probabilidade
    pred_class = np.argmax(prediction, axis=1)

    # Obtém o rótulo da classe prevista
    pred_label = labelencoder.inverse_transform(pred_class)

    # Obtém a confiança da predição
    confidence = prediction[0][pred_class][0]

    # Cria um dicionário com as probabilidades de cada classe
    class_probs = {labelencoder.classes_[i]: float(prediction[0][i]) for i in range(len(labelencoder.classes_))}

    return pred_label[0], confidence, class_probs

# ==================== CONFIGURAÇÃO DA APLICAÇÃO STREAMLIT ====================

def main():
    st.set_page_config(page_title="Classificação de Sons de Água Vibrando em Copo de Vidro", layout="wide")

    st.title("Classificação de Sons de Água Vibrando em Copo de Vidro com Aumento de Dados e CNN")
    st.write("""
    Esta aplicação permite classificar sons de água vibrando em copos de vidro. Você pode treinar um modelo CNN com seu próprio conjunto de dados ou utilizar um modelo pré-treinado para realizar previsões em novos arquivos de áudio.
    """)

    # Barra Lateral de Navegação
    st.sidebar.title("Navegação")
    app_mode = st.sidebar.selectbox("Escolha a seção", ["Classificar Áudio", "Treinar Modelo"])

    if app_mode == "Classificar Áudio":
        classificar_audio()
    elif app_mode == "Treinar Modelo":
        treinar_modelo()

def classificar_audio():
    st.header("Classificação de Novo Áudio")

    st.write("### Passo 1: Carregar o Modelo Treinado")
    st.write("**Formatos Aceitos:** `.keras`, `.h5` para modelos Keras ou `.pth` para modelos PyTorch.")
    modelo_file = st.file_uploader(
        "Faça upload do arquivo do modelo", 
        type=["keras", "h5", "pth"], 
        key="model_upload"
    )

    if modelo_file is not None:
        try:
            st.write(f"**Modelo Carregado:** {modelo_file.name}")
            # Salva o modelo temporariamente
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(modelo_file.name)[1]) as tmp_model:
                tmp_model.write(modelo_file.read())
                caminho_modelo = tmp_model.name

            # Carrega o modelo
            if caminho_modelo.endswith('.pth'):
                # Para modelos PyTorch, carregue de forma apropriada
                modelo = torch.load(caminho_modelo, map_location=torch.device('cpu'))
                modelo.eval()
                st.write("**Tipo de Modelo:** PyTorch")
            elif caminho_modelo.endswith(('.h5', '.keras')):
                # Para modelos Keras (.h5 e .keras)
                modelo = load_model(caminho_modelo, compile=False)
                st.write("**Tipo de Modelo:** Keras")
            else:
                st.error("Formato de modelo não suportado. Utilize .keras, .h5 ou .pth.")
                return
            st.success("Modelo carregado com sucesso!")

            # Carrega as classes
            st.write("### Passo 2: Carregar o Arquivo de Classes")
            st.write("**Formato Aceito:** `.txt`")
            classes_file = st.file_uploader(
                "Faça upload do arquivo com as classes (classes.txt)", 
                type=["txt"], 
                key="classes_upload"
            )
            if classes_file is not None:
                classes = classes_file.read().decode("utf-8").splitlines()
                labelencoder = LabelEncoder()
                labelencoder.fit(classes)
                st.success("Classes carregadas com sucesso!")
                st.write(f"**Classes:** {', '.join(classes)}")

                st.write("### Passo 3: Upload do Arquivo de Áudio para Classificação")
                audio_upload = st.file_uploader(
                    "Faça upload de um arquivo de áudio (.wav, .mp3, .flac, .ogg ou .m4a)", 
                    type=["wav", "mp3", "flac", "ogg", "m4a"], 
                    key="audio_upload"
                )

                if audio_upload is not None:
                    # Salva o arquivo de áudio temporariamente
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_upload.name)[1]) as tmp_audio:
                        tmp_audio.write(audio_upload.read())
                        caminho_audio = tmp_audio.name

                    # Exibe o áudio
                    st.audio(caminho_audio, format=f'audio/{os.path.splitext(audio_upload.name)[1][1:]}')

                    # Realiza a classificação
                    with st.spinner('Classificando...'):
                        rotulo_predito, confianca, probs_classes = processar_novo_audio(caminho_audio, modelo, labelencoder)

                    if rotulo_predito is not None and confianca is not None:
                        st.success(f"**Classe Predita:** {rotulo_predito}")
                        st.info(f"**Grau de Confiança:** {confianca * 100:.2f}%")

                        st.write("### Probabilidades das Classes:")
                        plot_probabilidades_classes(probs_classes, titulo="Probabilidades das Classes")

                        # Visualizações
                        st.write("### Visualizações do Áudio:")
                        data, sr = carregar_audio(caminho_audio, sr=None)
                        if data is not None:
                            plot_forma_onda(data, sr, titulo=f"Forma de Onda - {rotulo_predito}")
                            plot_espectro_frequencias(data, sr, titulo=f"Espectro de Frequências - {rotulo_predito}")
                            plot_espectrograma(data, sr, titulo=f"Espectrograma STFT - {rotulo_predito}")
                            plot_mfcc(data, sr, titulo=f"Espectrograma MFCC - {rotulo_predito}")
                    else:
                        st.error("A classificação não pôde ser realizada devido a erros no processamento do áudio.")

                    # Remove os arquivos temporários
                    os.remove(caminho_audio)
            # Remove o arquivo temporário do modelo
            os.remove(caminho_modelo)
        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {e}")
            # Assegura a remoção do arquivo temporário do modelo em caso de erro
            if 'caminho_modelo' in locals() and os.path.exists(caminho_modelo):
                os.remove(caminho_modelo)

    # **Passo 4: Não Existe na Seção de Classificação**
    # O arquivo classes.txt é baixado na seção "Treinar Modelo"

def treinar_modelo():
    st.header("Treinamento do Modelo CNN")

    st.write("""
    ### Passo 1: Upload do Dataset
    O dataset deve estar organizado em um arquivo ZIP com pastas para cada classe. Por exemplo:
    ```
    dataset.zip/
    ├── agua_gelada/
    │   ├── arquivo1.wav
    │   ├── arquivo2.wav
    │   └── ...
    ├── agua_quente/
    │   ├── arquivo1.wav
    │   ├── arquivo2.wav
    │   └── ...
    └── ...
    ```
    """)

    zip_upload = st.file_uploader(
        "Faça upload do arquivo ZIP contendo as pastas das classes", 
        type=["zip"], 
        key="dataset_upload"
    )

    if zip_upload is not None:
        try:
            st.write("#### Extraindo o Dataset...")
            # Salva o arquivo ZIP temporariamente
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                tmp_zip.write(zip_upload.read())
                caminho_zip = tmp_zip.name

            # Extrai o ZIP
            diretorio_extracao = tempfile.mkdtemp()
            with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
                zip_ref.extractall(diretorio_extracao)
            caminho_base = diretorio_extracao

            # Verifica se há subpastas (classes)
            categorias = [d for d in os.listdir(caminho_base) if os.path.isdir(os.path.join(caminho_base, d))]

            if len(categorias) == 0:
                st.error("Nenhuma subpasta de classes encontrada no ZIP. Verifique a estrutura do seu arquivo ZIP.")
                return

            st.success("Dataset extraído com sucesso!")
            st.write(f"**Classes encontradas:** {', '.join(categorias)}")

            # Coleta os caminhos dos arquivos e labels
            caminhos_arquivos = []
            labels = []
            for cat in categorias:
                caminho_cat = os.path.join(caminho_base, cat)
                arquivos_na_cat = [f for f in os.listdir(caminho_cat) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
                st.write(f"**Classe '{cat}':** {len(arquivos_na_cat)} arquivos encontrados.")
                if len(arquivos_na_cat) == 0:
                    st.warning(f"Nenhum arquivo encontrado na classe '{cat}'.")
                for nome_arquivo in arquivos_na_cat:
                    caminho_completo = os.path.join(caminho_cat, nome_arquivo)
                    caminhos_arquivos.append(caminho_completo)
                    labels.append(cat)

            df = pd.DataFrame({'caminho_arquivo': caminhos_arquivos, 'classe': labels})
            st.write("### Primeiras Amostras do Dataset:")
            st.dataframe(df.head())

            if len(df) == 0:
                st.error("Nenhuma amostra encontrada no dataset. Verifique os arquivos de áudio.")
                return

            # Codificação das classes
            labelencoder = LabelEncoder()
            y = labelencoder.fit_transform(df['classe'])
            classes = labelencoder.classes_
            st.write(f"**Classes codificadas:** {', '.join(classes)}")

            # **Exibir Número de Classes e Distribuição**
            st.write(f"### Número de Classes: {len(classes)}")
            contagem_classes = df['classe'].value_counts()
            st.write("### Distribuição das Classes:")
            fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
            sns.barplot(x=contagem_classes.index, y=contagem_classes.values, palette='viridis', ax=ax_dist)
            ax_dist.set_xlabel("Classes")
            ax_dist.set_ylabel("Número de Amostras")
            ax_dist.set_title("Distribuição das Classes no Dataset")
            st.pyplot(fig_dist)
            plt.close(fig_dist)

            # ==================== COLUNA DE CONFIGURAÇÃO ====================
            st.sidebar.header("Configurações de Treinamento")

            # Número de Épocas
            num_epochs = st.sidebar.slider(
                "Número de Épocas:",
                min_value=10,
                max_value=500,
                value=200,
                step=10,
                help="Define quantas vezes o modelo percorrerá todo o conjunto de dados durante o treinamento."
            )

            # Tamanho do Batch
            batch_size = st.sidebar.selectbox(
                "Tamanho do Batch:",
                options=[8, 16, 32, 64, 128],
                index=0,  # Seleciona 8 como padrão
                help="Número de amostras processadas antes de atualizar os pesos do modelo. Mínimo de 8."
            )

            # Fator de Aumento de Dados
            augment_factor = st.sidebar.slider(
                "Fator de Aumento de Dados:",
                min_value=1,
                max_value=20,
                value=10,
                step=1,
                help="Define quantas amostras aumentadas serão geradas a partir de cada amostra original."
            )

            # Taxa de Dropout
            dropout_rate = st.sidebar.slider(
                "Taxa de Dropout:",
                min_value=0.0,
                max_value=0.9,
                value=0.4,
                step=0.05,
                help="Proporção de neurônios a serem desligados durante o treinamento para evitar overfitting."
            )

            # Ativar/Desativar Data Augmentation
            enable_augmentation = st.sidebar.checkbox(
                "Ativar Data Augmentation",
                value=True,
                help="Permite ao usuário escolher se deseja ou não aplicar técnicas de aumento de dados."
            )

            # Seleção de Tipos de Data Augmentation
            if enable_augmentation:
                st.sidebar.subheader("Tipos de Data Augmentation")
                adicionar_ruido = st.sidebar.checkbox(
                    "Adicionar Ruído Gaussiano",
                    value=True,
                    help="Adiciona ruído gaussiano ao áudio para simular variações de som."
                )
                estiramento_tempo = st.sidebar.checkbox(
                    "Estiramento de Tempo",
                    value=True,
                    help="Altera a velocidade do áudio sem alterar seu pitch."
                )
                alteracao_pitch = st.sidebar.checkbox(
                    "Alteração de Pitch",
                    value=True,
                    help="Altera o pitch do áudio sem alterar sua velocidade."
                )
                deslocamento = st.sidebar.checkbox(
                    "Deslocamento",
                    value=True,
                    help="Desloca o áudio no tempo, adicionando silêncio no início ou no final."
                )

            # Balanceamento Ponderado das Classes
            st.sidebar.subheader("Balanceamento das Classes")
            balance_classes = st.sidebar.selectbox(
                "Método de Balanceamento das Classes:",
                options=["Balanced", "None"],
                index=0,
                help="Escolha 'Balanced' para aplicar balanceamento ponderado das classes ou 'None' para não aplicar."
            )
            # ==================== FIM DA COLUNA DE CONFIGURAÇÃO ====================

            # Extração de Features
            st.write("### Extraindo Features (MFCCs)...")
            X = []
            y_valid = []

            for i, row in df.iterrows():
                arquivo = row['caminho_arquivo']
                data, sr = carregar_audio(arquivo, sr=None)
                if data is not None:
                    features = extrair_features(data, sr)
                    if features is not None:
                        X.append(features)
                        y_valid.append(y[i])
                    else:
                        st.warning(f"Erro na extração de features do arquivo '{arquivo}'.")
                else:
                    st.warning(f"Erro no carregamento do arquivo '{arquivo}'.")

            X = np.array(X)
            y_valid = np.array(y_valid)

            st.write(f"**Features extraídas:** {X.shape}")

            # Divisão dos Dados
            st.write("### Dividindo os Dados em Treino e Teste...")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_valid, test_size=0.2, random_state=SEED, stratify=y_valid)
            st.write(f"**Treino:** {X_train.shape}, **Teste:** {X_test.shape}")

            # Data Augmentation no Treino
            if enable_augmentation:
                st.write("### Aplicando Data Augmentation no Conjunto de Treino...")
                X_train_augmented = []
                y_train_augmented = []

                for i in range(len(X_train)):
                    arquivo = df['caminho_arquivo'].iloc[i]
                    data, sr = carregar_audio(arquivo, sr=None)
                    if data is not None:
                        for _ in range(augment_factor):
                            # Aplicar apenas as transformações selecionadas
                            transformacoes = []
                            if adicionar_ruido:
                                transformacoes.append(AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0))
                            if estiramento_tempo:
                                transformacoes.append(TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0))
                            if alteracao_pitch:
                                transformacoes.append(PitchShift(min_semitones=-4, max_semitones=4, p=1.0))
                            if deslocamento:
                                transformacoes.append(Shift(min_shift=-0.5, max_shift=0.5, p=1.0))

                            if transformacoes:
                                augmentations = Compose(transformacoes)
                                augmented_data = aumentar_audio(data, sr, augmentations)

                                features = extrair_features(augmented_data, sr)
                                if features is not None:
                                    X_train_augmented.append(features)
                                    y_train_augmented.append(y_train[i])
                                else:
                                    st.warning(f"Erro na extração de features de uma amostra aumentada do arquivo '{arquivo}'.")
                    else:
                        st.warning(f"Erro no carregamento do arquivo '{arquivo}' para Data Augmentation.")

                X_train_augmented = np.array(X_train_augmented)
                y_train_augmented = np.array(y_train_augmented)
                st.write(f"**Dados aumentados:** {X_train_augmented.shape}")

                # Combinação dos Dados
                X_train_combined = np.concatenate((X_train, X_train_augmented), axis=0)
                y_train_combined = np.concatenate((y_train, y_train_augmented), axis=0)
                st.write(f"**Treino combinado:** {X_train_combined.shape}")
            else:
                X_train_combined = X_train
                y_train_combined = y_train

            # Divisão em Treino Final e Validação
            st.write("### Dividindo o Treino Combinado em Treino Final e Validação...")
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train_combined, y_train_combined, test_size=0.1, random_state=SEED, stratify=y_train_combined)
            st.write(f"**Treino Final:** {X_train_final.shape}, **Validação:** {X_val.shape}")

            # Ajuste da Forma dos Dados para a CNN (Conv1D)
            st.write("### Ajustando a Forma dos Dados para a CNN (Conv1D)...")
            X_train_final = X_train_final.reshape((X_train_final.shape[0], X_train_final.shape[1], 1))
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            st.write(f"**Shapes:** Treino Final: {X_train_final.shape}, Validação: {X_val.shape}, Teste: {X_test.shape}")

            # Cálculo de Class Weights
            st.write("### Calculando Class Weights para Balanceamento das Classes...")
            from sklearn.utils.class_weight import compute_class_weight
            if balance_classes == "Balanced":
                class_weights = compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(y_train_final),
                    y=y_train_final
                )
                class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
                st.write(f"**Pesos das Classes:** {class_weight_dict}")
            else:
                class_weight_dict = None
                st.write("**Balanceamento de classes não aplicado.**")

            # Definição da Arquitetura da CNN
            st.write("### Definindo a Arquitetura da Rede Neural Convolucional (CNN)...")
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input

            modelo = Sequential([
                Input(shape=(X_train_final.shape[1], 1)),
                Conv1D(64, kernel_size=10, activation='relu'),
                Dropout(dropout_rate),
                MaxPooling1D(pool_size=4),
                Conv1D(128, kernel_size=10, activation='relu', padding='same'),
                Dropout(dropout_rate),
                MaxPooling1D(pool_size=4),
                Flatten(),
                Dense(64, activation='relu'),
                Dropout(dropout_rate),
                Dense(len(classes), activation='softmax')
            ])

            # Compilação do Modelo
            modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # **Exibição do Resumo do Modelo como Tabela**
            st.write("### Resumo do Modelo:")
            resumo_modelo = []
            for layer in modelo.layers:
                nome_layer = layer.name
                tipo_layer = layer.__class__.__name__
                try:
                    forma_saida = layer.output_shape
                except AttributeError:
                    forma_saida = 'N/A'
                parametros = layer.count_params()
                resumo_modelo.append({
                    'Camada (Tipo)': f"{nome_layer} ({tipo_layer})",
                    'Forma de Saída': forma_saida,
                    'Parâmetros': f"{parametros:,}"
                })

            # Criação do DataFrame
            df_resumo = pd.DataFrame(resumo_modelo)

            # Adicionar total de parâmetros
            total_parametros = modelo.count_params()
            parametros_trainable = np.sum([layer.count_params() for layer in modelo.layers if layer.trainable])
            parametros_nao_trainable = total_parametros - parametros_trainable

            # Exibição da tabela
            st.dataframe(df_resumo)

            # Exibição dos totais
            st.write(f"**Total de parâmetros:** {total_parametros:,} ({total_parametros / 1e3:.2f} KB)")
            st.write(f"**Parâmetros treináveis:** {parametros_trainable:,} ({parametros_trainable / 1e3:.2f} KB)")
            st.write(f"**Parâmetros não treináveis:** {parametros_nao_trainable:,} ({parametros_nao_trainable / 1e3:.2f} KB)")

            # Definição dos Callbacks
            st.write("### Configurando Callbacks para o Treinamento...")
            from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
            diretorio_salvamento = 'modelos_salvos'
            if not os.path.exists(diretorio_salvamento):
                os.makedirs(diretorio_salvamento)
                st.write(f"**Diretório '{diretorio_salvamento}' criado para salvamento do modelo.**")
            else:
                st.write(f"**Diretório '{diretorio_salvamento}' já existe.**")

            # **Ajuste do `filepath` para garantir que a string está corretamente fechada e sem `save_format`**
            checkpointer = ModelCheckpoint(
                filepath=os.path.join(diretorio_salvamento, 'modelo_agua_aumentado.keras'),  # Pode usar .h5 se preferir
                monitor='val_loss',
                verbose=1,
                save_best_only=True
                # Removido 'save_format'
            )

            # Parâmetros de EarlyStopping
            st.sidebar.subheader("Parâmetros de EarlyStopping")
            es_monitor = st.sidebar.selectbox(
                "Monitorar:",
                options=["val_loss", "val_accuracy"],
                index=0,
                help="Métrica a ser monitorada para EarlyStopping."
            )
            es_patience = st.sidebar.slider(
                "Paciência (Épocas):",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="Número de épocas sem melhoria antes de interromper o treinamento."
            )
            es_mode = st.sidebar.selectbox(
                "Modo:",
                options=["min", "max"],
                index=0,
                help="Define se a métrica monitorada deve ser minimizada ou maximizada."
            )

            earlystop = EarlyStopping(
                monitor=es_monitor,
                patience=es_patience,
                restore_best_weights=True,
                mode=es_mode
            )

            # Treinamento do Modelo
            st.write("### Iniciando o Treinamento do Modelo...")
            with st.spinner('Treinando o modelo...'):
                historico = modelo.fit(
                    X_train_final, to_categorical(y_train_final),
                    epochs=num_epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, to_categorical(y_val)),
                    callbacks=[checkpointer, earlystop],
                    class_weight=class_weight_dict,
                    verbose=1
                )
            st.success("Treinamento concluído com sucesso!")

            # Salvamento do Modelo e Classes
            st.write("### Download do Modelo Treinado e Arquivo de Classes")
            # Salvar o modelo em um arquivo temporário com extensão .keras
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_model:
                modelo.save(tmp_model.name)
                caminho_tmp_model = tmp_model.name

            # Ler o modelo salvo e preparar para download
            with open(caminho_tmp_model, 'rb') as f:
                modelo_bytes = f.read()

            buffer = io.BytesIO(modelo_bytes)

            st.download_button(
                label="Download do Modelo Treinado (.keras)",
                data=buffer,
                file_name="modelo_agua_aumentado.keras",
                mime="application/octet-stream"
            )

            # Remove o arquivo temporário após o download
            os.remove(caminho_tmp_model)

            # Salvar as classes
            classes_str = "\n".join(classes)
            st.download_button(
                label="Download das Classes (classes.txt)",
                data=classes_str,
                file_name="classes.txt",
                mime="text/plain"
            )

            # Avaliação do Modelo
            st.write("### Avaliação do Modelo nos Conjuntos de Treino, Validação e Teste")
            score_train = modelo.evaluate(X_train_final, to_categorical(y_train_final), verbose=0)
            score_val = modelo.evaluate(X_val, to_categorical(y_val), verbose=0)
            score_test = modelo.evaluate(X_test, to_categorical(y_test), verbose=0)

            st.write(f"**Acurácia no Treino:** {score_train[1]*100:.2f}%")
            st.write(f"**Acurácia na Validação:** {score_val[1]*100:.2f}%")
            st.write(f"**Acurácia no Teste:** {score_test[1]*100:.2f}%")

            # Predições no Conjunto de Teste
            y_pred = modelo.predict(X_test)
            y_pred_classes = y_pred.argmax(axis=1)
            y_true = y_test  # y_test já está em formato inteiro

            # Matriz de Confusão com Seaborn
            cm = confusion_matrix(y_true, y_pred_classes, labels=range(len(classes)))
            cm_df = pd.DataFrame(cm, index=classes, columns=classes)
            fig_cm, ax_cm = plt.subplots(figsize=(12,8))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_title("Matriz de Confusão")
            ax_cm.set_xlabel("Classe Prevista")
            ax_cm.set_ylabel("Classe Real")
            st.pyplot(fig_cm)
            plt.close(fig_cm)

            # Relatório de Classificação com Seaborn
            report = classification_report(y_true, y_pred_classes, labels=range(len(classes)),
                                           target_names=classes, zero_division=0, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.write("### Relatório de Classificação:")
            st.dataframe(report_df)

            # Visualizações das Métricas de Treinamento com Seaborn
            st.write("### Visualizações das Métricas de Treinamento")
            historico_df = pd.DataFrame(historico.history)
            fig_loss, ax_loss = plt.subplots()
            sns.lineplot(data=historico_df[['loss', 'val_loss']], ax=ax_loss)
            ax_loss.set_title("Perda (Loss) durante o Treinamento")
            ax_loss.set_xlabel("Época")
            ax_loss.set_ylabel("Loss")
            st.pyplot(fig_loss)
            plt.close(fig_loss)

            fig_acc, ax_acc = plt.subplots()
            sns.lineplot(data=historico_df[['accuracy', 'val_accuracy']], ax=ax_acc)
            ax_acc.set_title("Acurácia durante o Treinamento")
            ax_acc.set_xlabel("Época")
            ax_acc.set_ylabel("Acurácia")
            st.pyplot(fig_acc)
            plt.close(fig_acc)

            # Limpeza de Memória
            del modelo, historico, historico_df
            gc.collect()

            st.success("Processo de Treinamento e Avaliação concluído!")

            # Remoção dos arquivos temporários
            os.remove(caminho_zip)
            for cat in categorias:
                caminho_cat = os.path.join(caminho_base, cat)
                for arquivo in os.listdir(caminho_cat):
                    os.remove(os.path.join(caminho_cat, arquivo))
                os.rmdir(caminho_cat)
            os.rmdir(caminho_base)

        except Exception as e:
            st.error(f"Erro durante o processamento do dataset: {e}")
            # Assegura a remoção dos arquivos temporários em caso de erro
            if 'caminho_zip' in locals() and os.path.exists(caminho_zip):
                os.remove(caminho_zip)
            if 'caminho_base' in locals() and os.path.exists(caminho_base):
                for cat in categorias:
                    caminho_cat = os.path.join(caminho_base, cat)
                    for arquivo in os.listdir(caminho_cat):
                        os.remove(os.path.join(caminho_cat, arquivo))
                    os.rmdir(caminho_cat)
                os.rmdir(caminho_base)

if __name__ == "__main__":
    main()

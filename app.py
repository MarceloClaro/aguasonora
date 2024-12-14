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
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
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
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import shap

# ==================== CONFIGURAÇÃO DE LOGGING ====================
# Configurar o logging para rastrear experimentos
logging.basicConfig(
    filename='experiment_logs.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ==================== CONFIGURAÇÃO DA PÁGINA ====================

# Definição do SEED
seed_options = list(range(0, 61, 2))  # [0, 2, 4, ..., 60]
default_seed = 42  # Valor padrão
if default_seed not in seed_options:
    seed_options.insert(0, default_seed)

# Definir a configuração da página **ANTES** de qualquer outra chamada do Streamlit
icon_path = "logo.png"  # Verifique se o arquivo logo.png está no diretório correto

if os.path.exists(icon_path):
    try:
        st.set_page_config(page_title="Geomaker", page_icon=icon_path, layout="wide")
        logging.info(f"Ícone {icon_path} carregado com sucesso.")
    except Exception as e:
        st.set_page_config(page_title="Geomaker", layout="wide")
        logging.warning(f"Erro ao carregar o ícone {icon_path}: {e}")
else:
    st.set_page_config(page_title="Geomaker", layout="wide")
    logging.warning(f"Ícone '{icon_path}' não encontrado, carregando sem favicon.")

# ==================== CONFIGURAÇÕES GERAIS NO SIDEBAR ====================

# Agora, todas as chamadas do Streamlit podem ocorrer após set_page_config()
st.sidebar.header("Configurações Gerais")

# Definição do SEED
seed_selection = st.sidebar.selectbox(
    "Escolha o valor do SEED:",
    options=seed_options,
    index=seed_options.index(default_seed) if default_seed in seed_options else 0,  # 42 como valor padrão
    help="Define a semente para reprodutibilidade dos resultados."
)
SEED = seed_selection  # Definindo a variável SEED

# Definir todas as sementes para reprodutibilidade
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(SEED)
logging.info(f"SEED definido para {SEED}.")

# Adicionar o expander com a explicação do SEED
with st.sidebar.expander("📖 Valor de SEED - Semente"):
    st.markdown("""
    ## **O Que é o SEED?**
    
    Imagine que você está jogando um jogo de tabuleiro onde, em cada turno, você precisa lançar um dado para determinar quantas casas avançar. Cada vez que você joga, o dado pode cair em um número diferente, tornando o jogo imprevisível e emocionante.
    
    Agora, suponha que você queira que esse jogo seja **repetível exatamente da mesma forma** todas as vezes que jogar. Para isso, você decide **anotar a sequência de números que o dado cai** em cada lançamento. Com essa sequência anotada, você pode "programar" o jogo para que, sempre que quiser, os lançamentos dos dados sigam essa mesma sequência, garantindo que o jogo tenha o mesmo resultado todas as vezes.
    
    Nesse exemplo:
    
    - **O "dado" representa o processo aleatório** no seu programa (como embaralhar cartas, escolher amostras aleatórias de dados, etc.).
    - **A "sequência anotada" é o SEED**.
    
    ## **Por Que Usar o SEED?**
    
    1. **Reprodutibilidade (Repetir o Mesmo Resultado):**
       - **Sem SEED:** Cada vez que você executa o programa, os resultados podem variar porque o computador está fazendo escolhas aleatórias diferentes.
       - **Com SEED:** Definindo um SEED, você garante que todas as escolhas aleatórias sejam as mesmas em cada execução. Assim, você obtém os **mesmos resultados** toda vez que rodar o programa.
    
    2. **Facilitar a Depuração:**
       - Quando algo não está funcionando como esperado, ter resultados consistentes (graças ao SEED) facilita identificar e corrigir problemas no código.
    
    3. **Comparar Modelos ou Experimentos:**
       - Se você está testando diferentes versões de um modelo ou realizando vários experimentos, usar o mesmo SEED garante que as condições iniciais sejam iguais, permitindo uma comparação justa entre eles.
    
    ## **Como Funciona na Prática?**
    
    No seu programa, o **valor de SEED** é simplesmente um número que você escolhe. Esse número é usado para "iniciar" o processo de geração de números aleatórios. Pense nisso como definir o ponto de partida para uma sequência que, a partir daí, sempre seguirá a mesma ordem.
    
    **Exemplo Simples:**
    
    - **SEED = 42**
      - Toda vez que você define o SEED como 42, as "escolhas aleatórias" do seu programa (como números gerados, posições embaralhadas, etc.) seguirão uma sequência específica que começa com 42.
      
    - **SEED = 100**
      - Com o SEED definido como 100, a sequência de escolhas aleatórias será diferente da sequência iniciada com 42, mas **sempre a mesma** quando SEED é 100.
    
    ## **Analogia do SEED**
    
    Pense no SEED como uma **chave para uma receita de bolo**:
    
    - **Sem a chave (SEED):** Cada vez que você faz o bolo, pode ter pequenas diferenças – talvez um pouco mais de farinha, outro pouco menos de açúcar.
    - **Com a chave (SEED):** Você pode refazer o bolo exatamente igual todas as vezes, garantindo que cada detalhe da receita seja seguido à risca.
    
    ## **Resumo Rápido**
    
    - **SEED** é um número que você define para tornar as escolhas aleatórias do seu programa **repetíveis e previsíveis**.
    - Usar SEED ajuda a **reproduzir resultados**, **facilitar a depuração** e **comparar experimentos** de maneira justa.
    - É como ter uma **receita precisa** ou uma **sequência de lançamentos de dados** que você pode repetir sempre que quiser.
    
    Espero que essa explicação ajude a esclarecer o que é o SEED e por que ele é útil no desenvolvimento de programas, especialmente em tarefas que envolvem processos aleatórios como aprendizado de máquina e análise de dados!
    """)

# ==================== LOGO E IMAGEM DE CAPA ====================

# Definir o caminho do ícone
icon_path = "logo.png"  # Verifique se o arquivo logo.png está no diretório correto

# Carrega e exibe a capa.png na página principal
capa_path = 'capa (2).png'
if os.path.exists(capa_path):
    try:
        st.image(
            capa_path, 
            caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', 
            use_container_width=True
        )
    except UnidentifiedImageError:
        st.warning(f"Imagem '{capa_path}' não pôde ser carregada ou está corrompida.")
else:
    st.warning(f"Imagem '{capa_path}' não encontrada.")

# Carregar o logotipo na barra lateral
logo_path = "logo.png"
if os.path.exists(logo_path):
    try:
        st.sidebar.image(logo_path, width=200, use_container_width=False)
    except UnidentifiedImageError:
        st.sidebar.text("Imagem do logotipo não pôde ser carregada ou está corrompida.")
else:
    st.sidebar.text("Imagem do logotipo não encontrada.")

# Título da Aplicação
st.title("Classificação de Sons de Água Vibrando em Copo de Vidro com Aumento de Dados e CNN")
st.write("""
Bem-vindo à nossa aplicação! Aqui, você pode **classificar sons de água vibrando em copos de vidro**. Você tem duas opções:
- **Classificar Áudio:** Use um modelo já treinado para identificar o som.
- **Treinar Modelo:** Treine seu próprio modelo com seus dados de áudio.
""")

# Barra Lateral de Navegação com Abas
st.sidebar.title("Navegação")
app_mode = st.sidebar.radio("Escolha a seção", ["Classificar Áudio", "Treinar Modelo"])

# Adicionando o ícone na barra lateral
eu_icon_path = "eu.ico"
if os.path.exists(eu_icon_path):
    try:
        st.sidebar.image(eu_icon_path, width=80, use_container_width=False)
    except UnidentifiedImageError:
        st.sidebar.text("Imagem do ícone 'eu.ico' não pôde ser carregada ou está corrompida.")
else:
    st.sidebar.text("Imagem do ícone 'eu.ico' não encontrada.")

st.sidebar.write("""
Produzido pelo:
    
Projeto Geomaker + IA 

https://doi.org/10.5281/zenodo.13910277

- Professor: Marcelo Claro.

Contatos: marceloclaro@gmail.com

Whatsapp: (88)98158-7145

Instagram: [marceloclaro.geomaker](https://www.instagram.com/marceloclaro.geomaker/)
""")

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
        logging.error(f"Erro ao carregar o áudio {caminho_arquivo}: {e}")
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
        logging.error(f"Erro ao extrair MFCC: {e}")
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
        logging.error(f"Erro ao aplicar Data Augmentation: {e}")
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
    ax.set_title(titulo, fontsize=16)
    ld.waveshow(data, sr=sr, ax=ax)
    ax.set_xlabel("Tempo (segundos)", fontsize=14)
    ax.set_ylabel("Amplitude", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
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
    ax.set_title(titulo, fontsize=16)
    ax.plot(freqs, fft, color='blue')
    ax.set_xlabel("Frequência (Hz)", fontsize=14)
    ax.set_ylabel("Amplitude", fontsize=14)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=12)
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
    ax.set_title(titulo, fontsize=16)
    
    # Usar 'time' e 'hz' para evitar erros
    mappable = ld.specshow(DB, sr=sr, x_axis='time', y_axis='hz', cmap='magma', ax=ax)
    
    cbar = plt.colorbar(mappable=mappable, ax=ax, format='%+2.0f dB')
    cbar.ax.set_ylabel("Intensidade (dB)", fontsize=14)
    
    # Personalizar rótulos dos eixos
    ax.set_xlabel("Tempo (segundos)", fontsize=14)
    ax.set_ylabel("Frequência (Hz)", fontsize=14)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    st.pyplot(fig)
    plt.close(fig)
    
    # Adicionar explicação para o usuário
    with st.expander("📖 Entenda o Espectrograma (STFT)"):
        st.markdown("""
        ### O que é um Espectrograma (STFT)?
        
        Um **Espectrograma** é uma representação visual do espectro de frequências de um sinal ao longo do tempo. Ele mostra como as frequências presentes no áudio mudam à medida que o tempo passa.

        - **Eixo X (Tempo):** Representa o tempo em segundos.
        - **Eixo Y (Frequência):** Representa a frequência em Hertz (Hz).
        - **Cores:** Indicam a intensidade (ou amplitude) das frequências. Cores mais claras representam frequências mais intensas.
        
        **Exemplo Visual:**
        ![Espectrograma](https://commons.wikimedia.org/wiki/File:Spectrogram-19thC.png#/media/File:Spectrogram-19thC.png)
        """)

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
    ax.set_title(titulo, fontsize=16)
    
    # Usar 'time' e 'mel' para evitar erros
    mappable = ld.specshow(mfccs_db, x_axis='time', y_axis='mel', cmap='Spectral', sr=sr, ax=ax)
    
    cbar = plt.colorbar(mappable=mappable, ax=ax, format='%+2.f dB')
    cbar.ax.set_ylabel("Intensidade (dB)", fontsize=14)
    
    # Personalizar rótulos dos eixos
    ax.set_xlabel("Tempo (segundos)", fontsize=14)
    ax.set_ylabel("Frequência Mel", fontsize=14)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    st.pyplot(fig)
    plt.close(fig)
    
    # Adicionar explicação para o usuário
    with st.expander("📖 Entenda o Espectrograma de MFCC"):
        st.markdown("""
        ### O que são MFCCs?
    
        **MFCCs (Mel-Frequency Cepstral Coefficients)** são características extraídas do áudio que representam a potência espectral em diferentes frequências na escala Mel, que é mais alinhada com a percepção humana de som.

        - **Eixo X (Tempo):** Representa o tempo em segundos.
        - **Eixo Y (Frequência Mel):** Representa a frequência na escala Mel.
        - **Cores:** Indicam a intensidade das frequências. Cores mais claras representam frequências mais intensas.
        
        **Por que usar MFCCs?**
        MFCCs são amplamente utilizados em reconhecimento de fala e classificação de áudio porque capturam as características essenciais do som de forma compacta e eficaz.
        
        **Exemplo Visual:**
        ![Espectrograma de MFCC](https://upload.wikimedia.org/wikipedia/commons/1/1c/Spectrogram_of_white_noise.svg)
        """)

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
    sns.barplot(x=classes, y=probs, palette='viridis', ax=ax)
    ax.set_title(titulo, fontsize=16)
    ax.set_xlabel("Classes", fontsize=14)
    ax.set_ylabel("Probabilidade", fontsize=14)
    ax.set_ylim(0, 1)  # Probabilidades entre 0 e 1
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Adiciona rótulos de porcentagem acima das barras
    for i, v in enumerate(probs):
        ax.text(i, v + 0.01, f"{v*100:.2f}%", ha='center', va='bottom', fontsize=12)
    
    st.pyplot(fig)
    plt.close(fig)

def plot_roc_curve(y_true, y_score, classes):
    """
    Plota a curva ROC para cada classe.

    Parameters:
    - y_true (np.ndarray): Verdadeiros rótulos.
    - y_score (np.ndarray): Probabilidades preditas.
    - classes (list): Lista de nomes das classes.
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle

    y_true_binarized = label_binarize(y_true, classes=range(len(classes)))
    n_classes = y_true_binarized.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Curvas ROC das Classes', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    st.pyplot(plt.gcf())
    plt.close()

def plot_precision_recall_curve_custom(y_true, y_score, classes):
    """
    Plota a curva Precision-Recall para cada classe.

    Parameters:
    - y_true (np.ndarray): Verdadeiros rótulos.
    - y_score (np.ndarray): Probabilidades preditas.
    - classes (list): Lista de nomes das classes.
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from itertools import cycle

    y_true_binarized = label_binarize(y_true, classes=range(len(classes)))
    n_classes = y_true_binarized.shape[1]

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_true_binarized[:, i], y_score[:, i])

    # Plot all Precision-Recall curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'green', 'purple', 'brown', 'pink'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='Precision-Recall curve of class {0} (AP = {1:0.2f})'
                 ''.format(classes[i], average_precision[i]))

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Curvas Precision-Recall das Classes', fontsize=16)
    plt.legend(loc="lower left", fontsize=12)
    st.pyplot(plt.gcf())
    plt.close()

def plot_shap_values(model, X_sample, feature_names):
    """
    Plota os valores SHAP para explicar as previsões do modelo.

    Parameters:
    - model (tf.keras.Model): Modelo treinado.
    - X_sample (np.ndarray): Amostra de dados para explicação.
    - feature_names (list): Lista de nomes das features.
    """
    try:
        # Criar um objeto explainer SHAP
        explainer = shap.DeepExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)

        # Plot summary
        st.subheader("Explicação das Previsões com SHAP")
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        st.pyplot(bbox_inches='tight')
        plt.close()
    except Exception as e:
        st.error(f"Erro ao gerar explicações SHAP: {e}")
        logging.error(f"Erro ao gerar explicações SHAP: {e}")

def processar_novo_audio(caminho_audio, modelo, labelencoder):
    """
    Carrega, extrai features e classifica um novo arquivo de áudio.

    Parameters:
    - caminho_audio (str): Caminho para o arquivo de áudio.
    - modelo (tf.keras.Model): Modelo treinado para classificação.
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
    try:
        prediction = modelo.predict(mfccs)
    except Exception as e:
        st.error(f"Erro na predição: {e}")
        logging.error(f"Erro na predição: {e}")
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

def classificar_audio(SEED):
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
                logging.info("Modelo PyTorch carregado.")
            elif caminho_modelo.endswith(('.h5', '.keras')):
                # Para modelos Keras (.h5 e .keras)
                modelo = load_model(caminho_modelo, compile=False)
                st.write("**Tipo de Modelo:** Keras")
                logging.info("Modelo Keras carregado.")
            else:
                st.error("Formato de modelo não suportado. Utilize .keras, .h5 ou .pth.")
                logging.error("Formato de modelo não suportado.")
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
                logging.info(f"Classes carregadas: {', '.join(classes)}")

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

                        # Explicabilidade com SHAP
                        st.write("### Explicabilidade das Previsões com SHAP")
                        # Selecionar uma amostra do conjunto de treino para o explainer
                        if 'X_train_final' in st.session_state and st.session_state.X_train_final is not None:
                            X_sample = st.session_state.X_train_final[:100]  # Limitar a 100 amostras para performance
                        else:
                            # Se não houver, usar a própria amostra
                            X_sample = np.expand_dims(mfccs, axis=0)
                        plot_shap_values(modelo, X_sample, feature_names=[f'MFCC_{i}' for i in range(1, 41)])
                    else:
                        st.error("A classificação não pôde ser realizada devido a erros no processamento do áudio.")

                    # Remove os arquivos temporários
                    os.remove(caminho_audio)
                    os.remove(caminho_modelo)
        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {e}")
            logging.error(f"Erro ao carregar o modelo: {e}")
            # Assegura a remoção do arquivo temporário do modelo em caso de erro
            if 'caminho_modelo' in locals() and os.path.exists(caminho_modelo):
                os.remove(caminho_modelo)

def treinar_modelo(SEED):
    st.header("Treinamento do Modelo CNN")

    st.write("""
    ### Passo 1: Upload do Dataset
    O **dataset** deve estar organizado em um arquivo ZIP com pastas para cada classe. Por exemplo:
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
                logging.error("Nenhuma subpasta de classes encontrada no ZIP.")
                return

            st.success("Dataset extraído com sucesso!")
            st.write(f"**Classes encontradas:** {', '.join(categorias)}")
            logging.info(f"Classes encontradas: {', '.join(categorias)}")

            # Coleta os caminhos dos arquivos e labels
            caminhos_arquivos = []
            labels = []
            for cat in categorias:
                caminho_cat = os.path.join(caminho_base, cat)
                arquivos_na_cat = [f for f in os.listdir(caminho_cat) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
                st.write(f"**Classe '{cat}':** {len(arquivos_na_cat)} arquivos encontrados.")
                if len(arquivos_na_cat) == 0:
                    st.warning(f"Nenhum arquivo encontrado na classe '{cat}'.")
                    logging.warning(f"Nenhum arquivo encontrado na classe '{cat}'.")
                for nome_arquivo in arquivos_na_cat:
                    caminho_completo = os.path.join(caminho_cat, nome_arquivo)
                    caminhos_arquivos.append(caminho_completo)
                    labels.append(cat)

            df = pd.DataFrame({'caminho_arquivo': caminhos_arquivos, 'classe': labels})
            st.write("### Primeiras Amostras do Dataset:")
            st.dataframe(df.head())

            if len(df) == 0:
                st.error("Nenhuma amostra encontrada no dataset. Verifique os arquivos de áudio.")
                logging.error("Nenhuma amostra encontrada no dataset.")
                return

            # Codificação das classes
            labelencoder = LabelEncoder()
            y = labelencoder.fit_transform(df['classe'])
            classes = labelencoder.classes_
            st.write(f"**Classes codificadas:** {', '.join(classes)}")
            logging.info(f"Classes codificadas: {', '.join(classes)}")

            # **Explicação dos Dados**
            with st.expander("📖 Explicação dos Dados"):
                st.markdown("""
                ### Explicação dos Dados

                **1. Features Extraídas: (N, 40)**
                - **O que são Features?**
                  Features são características ou informações específicas extraídas dos dados brutos (neste caso, arquivos de áudio) que são usadas para treinar o modelo.
                - **Interpretação de (N, 40):**
                  - **N:** Número de amostras ou exemplos no conjunto de dados.
                  - **40:** Número de características extraídas de cada amostra.
                - **Explicação Simples:**
                  Cada arquivo de áudio tem 40 características (MFCCs) extraídas, representando aspectos importantes do som para o modelo aprender.

                **2. Divisão dos Dados:**
                Após extrair as features, os dados são divididos em diferentes conjuntos para treinar e avaliar o modelo.
                """)

            # **Exibir Número de Classes e Distribuição**
            st.write(f"### Número de Classes: {len(classes)}")
            contagem_classes = df['classe'].value_counts()
            st.write("### Distribuição das Classes:")
            fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
            sns.barplot(x=contagem_classes.index, y=contagem_classes.values, palette='viridis', ax=ax_dist, legend=False)
            ax_dist.set_xlabel("Classes", fontsize=14)
            ax_dist.set_ylabel("Número de Amostras", fontsize=14)
            ax_dist.set_title("Distribuição das Classes no Dataset", fontsize=16)
            ax_dist.tick_params(axis='both', which='major', labelsize=12)
            st.pyplot(fig_dist)
            plt.close(fig_dist)
            logging.info("Distribuição das classes exibida.")

            # ==================== CONFIGURAÇÕES DE TREINAMENTO ====================
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

            # Percentual de Divisão Treino/Teste/Validação
            st.sidebar.subheader("Divisão dos Dados")
            treino_percentage = st.sidebar.slider(
                "Percentual para o Conjunto de Treino (%)",
                min_value=50,
                max_value=90,
                value=70,
                step=5,
                help="Define a porcentagem dos dados que serão usados para o conjunto de treino."
            )
            valid_percentage = st.sidebar.slider(
                "Percentual para o Conjunto de Validação (%)",
                min_value=5,
                max_value=30,
                value=15,
                step=5,
                help="Define a porcentagem dos dados que serão usados para o conjunto de validação."
            )
            # Calcula o percentual para o teste
            test_percentage = 100 - (treino_percentage + valid_percentage)
            if test_percentage < 0:
                st.sidebar.error("A soma dos percentuais de treino e validação excede 100%. Ajuste os valores.")
                logging.error("Percentual de treino + validação > 100%.")
                st.stop()
            st.sidebar.write(f"**Percentual para o Conjunto de Teste:** {test_percentage}%")

            # Fator de Aumento de Dados
            augment_factor = st.sidebar.slider(
                "Fator de Aumento de Dados:",
                min_value=1,
                max_value=100,
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

            # Taxa de Regularização L1 e L2
            st.sidebar.subheader("Regularização")
            regularization_type = st.sidebar.selectbox(
                "Tipo de Regularização:",
                options=["None", "L1", "L2", "L1_L2"],
                index=0,
                help="Escolha o tipo de regularização a ser aplicada nas camadas do modelo."
            )
            if regularization_type == "L1":
                l1_regularization = st.sidebar.slider(
                    "Taxa de Regularização L1:",
                    min_value=0.0,
                    max_value=0.1,
                    value=0.001,
                    step=0.001,
                    help="Define a taxa de regularização L1 para evitar overfitting."
                )
                l2_regularization = 0.0
            elif regularization_type == "L2":
                l2_regularization = st.sidebar.slider(
                    "Taxa de Regularização L2:",
                    min_value=0.0,
                    max_value=0.1,
                    value=0.001,
                    step=0.001,
                    help="Define a taxa de regularização L2 para evitar overfitting."
                )
                l1_regularization = 0.0
            elif regularization_type == "L1_L2":
                l1_regularization = st.sidebar.slider(
                    "Taxa de Regularização L1:",
                    min_value=0.0,
                    max_value=0.1,
                    value=0.001,
                    step=0.001,
                    help="Define a taxa de regularização L1 para evitar overfitting."
                )
                l2_regularization = st.sidebar.slider(
                    "Taxa de Regularização L2:",
                    min_value=0.0,
                    max_value=0.1,
                    value=0.001,
                    step=0.001,
                    help="Define a taxa de regularização L2 para evitar overfitting."
                )
            else:
                l1_regularization = 0.0
                l2_regularization = 0.0

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
                    help="Altera a velocidade do áudio sem alterar seu tom."
                )
                alteracao_pitch = st.sidebar.checkbox(
                    "Alteração de Pitch",
                    value=True,
                    help="Altera o tom do áudio sem alterar sua velocidade."
                )
                deslocamento = st.sidebar.checkbox(
                    "Deslocamento",
                    value=True,
                    help="Desloca o áudio no tempo, adicionando silêncio no início ou no final."
                )

            # Opções de Cross-Validation
            st.sidebar.subheader("Validação Cruzada")
            cross_validation = st.sidebar.checkbox(
                "Ativar Validação Cruzada (k-Fold)",
                value=False,
                help="Ativa a validação cruzada para uma avaliação mais robusta do modelo."
            )
            if cross_validation:
                k_folds = st.sidebar.number_input(
                    "Número de Folds:",
                    min_value=2,
                    max_value=10,
                    value=5,
                    step=1,
                    help="Define o número de folds para a validação cruzada."
                )
            else:
                k_folds = 1  # Não utilizado

            # Balanceamento Ponderado das Classes
            st.sidebar.subheader("Balanceamento das Classes")
            balance_classes = st.sidebar.selectbox(
                "Método de Balanceamento das Classes:",
                options=["Balanced", "None"],
                index=0,
                help="Escolha 'Balanced' para aplicar balanceamento ponderado das classes ou 'None' para não aplicar."
            )
            logging.info("Configurações de treinamento definidas pelo usuário.")

            # ==================== FIM DA CONFIGURAÇÃO DE TREINAMENTO ====================

            # Extração de Features
            st.write("### Extraindo Features (MFCCs)...")
            st.write("""
            **MFCCs (Mel-Frequency Cepstral Coefficients)** são características extraídas do áudio que representam a potência espectral em diferentes frequências. Eles são amplamente utilizados em processamento de áudio e reconhecimento de padrões, pois capturam informações relevantes para identificar sons distintos.
            """)
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
                        logging.warning(f"Erro na extração de features do arquivo '{arquivo}'.")
                else:
                    st.warning(f"Erro no carregamento do arquivo '{arquivo}'.")
                    logging.warning(f"Erro no carregamento do arquivo '{arquivo}'.")

            X = np.array(X)
            y_valid = np.array(y_valid)

            st.write(f"**Features extraídas:** {X.shape}")
            logging.info(f"Features extraídas: {X.shape}")

            # **Explicação das Features Extraídas**
            with st.expander("📖 Explicação das Features Extraídas"):
                st.markdown("""
                **1. Features Extraídas: (N, 40)**
                - **O que são Features?**
                  Features são características ou informações específicas extraídas dos dados brutos (neste caso, arquivos de áudio) que são usadas para treinar o modelo.
                - **Interpretação de (N, 40):**
                  - **N:** Número de amostras ou exemplos no conjunto de dados.
                  - **40:** Número de características extraídas de cada amostra.
                - **Explicação Simples:**
                  Cada arquivo de áudio tem 40 características (MFCCs) extraídas, representando aspectos importantes do som para o modelo aprender.
                """)

            # Divisão dos Dados
            st.write("### Dividindo os Dados em Treino, Validação e Teste...")
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y_valid, test_size=(100 - treino_percentage)/100.0, random_state=SEED, stratify=y_valid)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=test_percentage/(test_percentage + valid_percentage), random_state=SEED, stratify=y_temp)
            st.write(f"**Treino:** {X_train.shape}, **Validação:** {X_val.shape}, **Teste:** {X_test.shape}")
            logging.info(f"Divisão dos dados: Treino={X_train.shape}, Validação={X_val.shape}, Teste={X_test.shape}")

            # **Explicação da Divisão dos Dados**
            with st.expander("📖 Explicação da Divisão dos Dados"):
                st.markdown("""
                **2. Divisão dos Dados:**
                Após extrair as features, os dados são divididos em diferentes conjuntos para treinar e avaliar o modelo.

                - **Treino: (N_train, 40)**
                  - **N_train:** Número de amostras usadas para treinar o modelo.
                  - **40:** Número de características por amostra.
                  - **Explicação:** Uma porcentagem definida pelo usuário é usada para treinar o modelo.

                - **Validação: (N_val, 40)**
                  - **N_val:** Número de amostras usadas para validar o modelo durante o treinamento.
                  - **40:** Número de características por amostra.
                  - **Explicação:** Uma porcentagem definida pelo usuário é usada para validar o modelo e ajustar hiperparâmetros.

                - **Teste: (N_test, 40)**
                  - **N_test:** Número de amostras usadas para testar a performance do modelo.
                  - **40:** Número de características por amostra.
                  - **Explicação:** A porcentagem restante é usada para avaliar o modelo após o treinamento.
                """)

            # Data Augmentation no Treino
            if enable_augmentation:
                st.write("### Aplicando Data Augmentation no Conjunto de Treino...")
                st.write("""
                **Data Augmentation** é uma técnica utilizada para aumentar a quantidade e diversidade dos dados de treinamento aplicando transformações nos dados originais. Isso ajuda o modelo a generalizar melhor e reduzir o overfitting.
                """)
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
                                    logging.warning(f"Erro na extração de features de uma amostra aumentada do arquivo '{arquivo}'.")
                    else:
                        st.warning(f"Erro no carregamento do arquivo '{arquivo}' para Data Augmentation.")
                        logging.warning(f"Erro no carregamento do arquivo '{arquivo}' para Data Augmentation.")

                X_train_augmented = np.array(X_train_augmented)
                y_train_augmented = np.array(y_train_augmented)
                st.write(f"**Dados aumentados:** {X_train_augmented.shape}")
                logging.info(f"Dados aumentados: {X_train_augmented.shape}")

                # **Explicação dos Dados Aumentados**
                with st.expander("📖 Explicação dos Dados Aumentados"):
                    st.markdown("""
                    **Dados Aumentados: (N_augmented, 40)**
                    - **N_augmented:** Número de amostras adicionais geradas através de técnicas de aumento de dados.
                    - **40:** Número de características por amostra.
                    - **Explicação:** Para melhorar a performance do modelo, criamos novas amostras a partir das originais, aplicando transformações como adicionar ruído, alterar o pitch, estirar o tempo ou deslocar o áudio.
                    """)

                # Combinação dos Dados
                X_train_combined = np.concatenate((X_train, X_train_augmented), axis=0)
                y_train_combined = np.concatenate((y_train, y_train_augmented), axis=0)
                st.write(f"**Treino combinado:** {X_train_combined.shape}")
                logging.info(f"Treino combinado: {X_train_combined.shape}")

                # **Explicação da Combinação dos Dados**
                with st.expander("📖 Explicação da Combinação dos Dados"):
                    st.markdown("""
                    **3. Combinação e Validação:**
                    - **Treino Combinado: (N_train + N_augmented, 40)**
                      - **N_train + N_augmented:** Soma das amostras de treino original e aumentadas.
                      - **40:** Número de características por amostra.
                      - **Explicação:** Unimos as amostras originais com as aumentadas para formar um conjunto de treino mais robusto.
                    """)

            else:
                X_train_combined = X_train
                y_train_combined = y_train
                logging.info("Data Augmentation desativado.")

            # Divisão em Treino Final e Validação (já realizada anteriormente)
            st.write("### Dividindo o Treino Combinado em Treino Final e Validação...")
            if cross_validation:
                st.write("**Validação Cruzada Ativada.** O conjunto de validação será gerado durante o processo de cross-validation.")
                logging.info("Validação cruzada ativada.")
            else:
                # Divisão adicional para validação
                X_train_final, X_val, y_train_final, y_val = train_test_split(
                    X_train_combined, y_train_combined, test_size=0.1, random_state=SEED, stratify=y_train_combined)
                st.write(f"**Treino Final:** {X_train_final.shape}, **Validação:** {X_val.shape}")
                logging.info(f"Treino Final: {X_train_final.shape}, Validação: {X_val.shape}")

                # **Explicação da Divisão Final**
                with st.expander("📖 Explicação da Divisão Final"):
                    st.markdown("""
                    **3. Combinação e Validação:**
                    - **Treino Combinado: (N_train + N_augmented, 40)**
                      - **N_train + N_augmented:** Soma das amostras de treino original e aumentadas.
                      - **40:** Número de características por amostra.
                      - **Explicação:** Unimos as amostras originais com as aumentadas para formar um conjunto de treino mais robusto.
                    - **Treino Final: (N_final, 40)**
                      - **N_final:** Número de amostras após uma divisão adicional para validação.
                      - **40:** Número de características por amostra.
                      - **Explicação:** Das amostras combinadas, uma parte é usada para treinar o modelo definitivamente.
                    - **Validação: (N_val, 40)**
                      - **N_val:** Número de amostras usadas para validar o modelo durante o treinamento.
                      - **40:** Número de características por amostra.
                      - **Explicação:** As amostras restantes são usadas para monitorar se o modelo está aprendendo de forma adequada.
                    """)

            # Ajuste da Forma dos Dados para a CNN (Conv1D)
            st.write("### Ajustando a Forma dos Dados para a CNN (Conv1D)...")
            if cross_validation:
                # Para cross-validation, manter a forma original
                st.write("**Cross-Validation Ativado:** A forma dos dados será ajustada durante o treinamento.")
                logging.info("Forma dos dados para CNN será ajustada durante a validação cruzada.")
            else:
                X_train_final = X_train_final.reshape((X_train_final.shape[0], X_train_final.shape[1], 1))
                X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                st.write(f"**Shapes:** Treino Final: {X_train_final.shape}, Validação: {X_val.shape}, Teste: {X_test.shape}")
                logging.info(f"Shapes ajustadas: Treino Final: {X_train_final.shape}, Validação: {X_val.shape}, Teste: {X_test.shape}")

            # Cálculo de Class Weights
            st.write("### Calculando Class Weights para Balanceamento das Classes...")
            st.write("""
            **Class Weights** são utilizados para lidar com desequilíbrios nas classes do conjunto de dados. Quando algumas classes têm muito mais amostras do que outras, o modelo pode se tornar tendencioso em favor das classes mais frequentes. Aplicar pesos balanceados ajuda o modelo a prestar mais atenção às classes menos representadas.
            """)
            if balance_classes == "Balanced":
                if cross_validation:
                    st.warning("Balanceamento de classes durante Cross-Validation não está implementado.")
                    class_weight_dict = None
                    logging.warning("Balanceamento de classes não implementado para Cross-Validation.")
                else:
                    class_weights = compute_class_weight(
                        class_weight='balanced',
                        classes=np.unique(y_train_final),
                        y=y_train_final
                    )
                    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
                    st.write(f"**Pesos das Classes:** {class_weight_dict}")
                    logging.info(f"Pesos das classes calculados: {class_weight_dict}")
            else:
                class_weight_dict = None
                st.write("**Balanceamento de classes não aplicado.**")
                logging.info("Balanceamento de classes não aplicado.")

            # Definição da Arquitetura da CNN com Regularização
            st.write("### Definindo a Arquitetura da Rede Neural Convolucional (CNN)...")
            st.write("""
            A **Rede Neural Convolucional (CNN)** é uma arquitetura de rede neural eficaz para processamento de dados com estrutura de grade, como imagens e sinais de áudio. Nesta aplicação, utilizamos camadas convolucionais para extrair características relevantes dos dados de áudio.

            **Personalize a Arquitetura:**
            Você pode ajustar os seguintes hiperparâmetros:
            - **Número de Camadas Convolucionais**
            - **Número de Filtros por Camada**
            - **Tamanho do Kernel**
            - **Tipo e Taxa de Regularização (L1, L2 ou ambas)**
            """)

            # Hiperparâmetros da CNN
            st.sidebar.subheader("Arquitetura da CNN")

            num_conv_layers = st.sidebar.slider(
                "Número de Camadas Convolucionais:",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                help="Define o número de camadas convolucionais na rede."
            )

            conv_filters = st.sidebar.text_input(
                "Número de Filtros por Camada (Separados por vírgula):",
                value="64,128",
                help="Defina o número de filtros para cada camada convolucional, separados por vírgula. Exemplo: 64,128"
            )

            conv_kernel_size = st.sidebar.text_input(
                "Tamanho do Kernel por Camada (Separados por vírgula):",
                value="10,10",
                help="Defina o tamanho do kernel para cada camada convolucional, separados por vírgula. Exemplo: 10,10"
            )

            # Processar as entradas de filtros e tamanho do kernel
            try:
                conv_filters = [int(f.strip()) for f in conv_filters.split(',')]
                conv_kernel_size = [int(k.strip()) for k in conv_kernel_size.split(',')]
                if len(conv_filters) != num_conv_layers or len(conv_kernel_size) != num_conv_layers:
                    st.sidebar.error("O número de filtros e tamanhos de kernel deve corresponder ao número de camadas convolucionais.")
                    logging.error("Número de filtros e tamanhos de kernel não corresponde ao número de camadas.")
                    st.stop()
            except ValueError:
                st.sidebar.error("Certifique-se de que os filtros e tamanhos de kernel sejam números inteiros separados por vírgula.")
                logging.error("Erro na conversão de filtros ou tamanhos de kernel para inteiros.")
                st.stop()

            # Número de Camadas Densas
            st.sidebar.subheader("Arquitetura da CNN - Camadas Densas")
            num_dense_layers = st.sidebar.slider(
                "Número de Camadas Densas:",
                min_value=1,
                max_value=3,
                value=1,
                step=1,
                help="Define o número de camadas densas na rede."
            )

            dense_units = st.sidebar.text_input(
                "Número de Neurônios por Camada Densa (Separados por vírgula):",
                value="64",
                help="Defina o número de neurônios para cada camada densa, separados por vírgula. Exemplo: 64,32"
            )

            try:
                dense_units = [int(u.strip()) for u in dense_units.split(',')]
                if len(dense_units) != num_dense_layers:
                    st.sidebar.error("O número de neurônios deve corresponder ao número de camadas densas.")
                    logging.error("Número de neurônios não corresponde ao número de camadas densas.")
                    st.stop()
            except ValueError:
                st.sidebar.error("Certifique-se de que os neurônios sejam números inteiros separados por vírgula.")
                logging.error("Erro na conversão de neurônios para inteiros.")
                st.stop()

            # Definição da Arquitetura da CNN com Regularização
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input

            modelo = Sequential()
            modelo.add(Input(shape=(X_train_combined.shape[1], 1)))

            # Adicionar Camadas Convolucionais
            for i in range(num_conv_layers):
                if regularization_type == "L1":
                    reg = regularizers.l1(l1_regularization)
                elif regularization_type == "L2":
                    reg = regularizers.l2(l2_regularization)
                elif regularization_type == "L1_L2":
                    reg = regularizers.l1_l2(l1=l1_regularization, l2=l2_regularization)
                else:
                    reg = None

                modelo.add(Conv1D(
                    filters=conv_filters[i],
                    kernel_size=conv_kernel_size[i],
                    activation='relu',
                    kernel_regularizer=reg
                ))
                modelo.add(Dropout(dropout_rate))
                modelo.add(MaxPooling1D(pool_size=4))

            modelo.add(Flatten())

            # Adicionar Camadas Densas
            for i in range(num_dense_layers):
                if regularization_type in ["L1", "L2", "L1_L2"]:
                    reg = regularizers.l1_l2(l1=l1_regularization, l2=l2_regularization) if regularization_type == "L1_L2" else (regularizers.l1(l1_regularization) if regularization_type == "L1" else regularizers.l2(l2_regularization))
                else:
                    reg = None

                modelo.add(Dense(
                    units=dense_units[i],
                    activation='relu',
                    kernel_regularizer=reg
                ))
                modelo.add(Dropout(dropout_rate))

            # Camada de Saída
            modelo.add(Dense(len(classes), activation='softmax'))

            # Compilação do Modelo
            modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            logging.info("Modelo compilado.")

            # **Exibição do Resumo do Modelo como Tabela**
            st.write("### Resumo do Modelo:")
            st.write("""
            A tabela abaixo apresenta as camadas da rede neural, a forma de saída de cada camada e o número de parâmetros (pesos) que cada camada possui. 
            - **Camada (Tipo):** Nome e tipo da camada.
            - **Forma de Saída:** Dimensões da saída da camada.
            - **Parâmetros:** Número de parâmetros treináveis na camada.
            """)

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
            logging.info("Resumo do modelo exibido.")

            # **Explicação das Camadas do Modelo**
            with st.expander("📖 Explicação das Camadas do Modelo"):
                st.markdown("""
                ### Explicação das Camadas do Modelo

                As camadas de uma Rede Neural Convolucional (CNN) são responsáveis por processar e aprender padrões nos dados. Vamos explicar cada uma das camadas presentes no seu modelo de forma simples:

                **1. Conv1D (Conv1D)**
                - **O que é?**
                  Conv1D é uma camada convolucional unidimensional usada para processar dados sequenciais, como áudio ou séries temporais.
                - **Função:**
                  **Extrair Padrões Locais:** Ela passa uma janela (filtro) sobre os dados para detectar padrões específicos, como certas frequências ou ritmos no áudio.
                - **Regularização:**
                  Dependendo da configuração, aplica regularização L1, L2 ou ambas para evitar overfitting.
                - **Exemplo no Modelo:**
                  ```python
                  Conv1D(64, kernel_size=10, activation='relu', kernel_regularizer=regularizers.l2(0.001))
                  ```
                  - **64:** Número de filtros (detetores de padrões) usados.
                  - **kernel_size=10:** Tamanho da janela que percorre os dados.
                  - **activation='relu':** Função de ativação que introduz não-linearidade.
                  - **kernel_regularizer=regularizers.l2(0.001):** Aplicação da regularização L2 com taxa 0.001.

                **2. Dropout (Dropout)**
                - **O que é?**
                  Dropout é uma técnica de regularização que ajuda a prevenir o overfitting.
                - **Função:**
                  **Desligar Neurônios Aleatoriamente:** Durante o treinamento, desliga aleatoriamente uma porcentagem dos neurônios, forçando o modelo a não depender excessivamente de nenhum neurônio específico.
                - **Exemplo no Modelo:**
                  ```python
                  Dropout(0.4)
                  ```
                  - **0.4:** 40% dos neurônios serão desligados durante o treinamento.

                **3. MaxPooling1D (MaxPooling1D)**
                - **O que é?**
                  MaxPooling1D é uma camada de pooling que reduz a dimensionalidade dos dados.
                - **Função:**
                  **Reduzir a Dimensionalidade:** Seleciona o valor máximo em cada janela de tamanho especificado, resumindo a informação e reduzindo o número de parâmetros.
                - **Exemplo no Modelo:**
                  ```python
                  MaxPooling1D(pool_size=4)
                  ```
                  - **pool_size=4:** Seleciona o maior valor em janelas de 4 unidades.

                **4. Flatten (Flatten)**
                - **O que é?**
                  Flatten é uma camada que transforma os dados multidimensionais em um vetor unidimensional.
                - **Função:**
                  **Preparar para Camadas Densas:** Converte a saída das camadas convolucionais em uma forma adequada para as camadas densas (totalmente conectadas).
                - **Exemplo no Modelo:**
                  ```python
                  Flatten()
                  ```
                  - Sem parâmetros, apenas altera a forma dos dados.

                **5. Dense (Dense)**
                - **O que é?**
                  Dense é uma camada totalmente conectada onde cada neurônio está conectado a todos os neurônios da camada anterior.
                - **Função:**
                  **Tomar Decisões Finais:** Combina todas as características extraídas pelas camadas anteriores para fazer a classificação final.
                - **Regularização:**
                  Dependendo da configuração, aplica regularização L1, L2 ou ambas para evitar overfitting.
                - **Exemplo no Modelo:**
                  ```python
                  Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))
                  ```
                  - **64:** Número de neurônios na camada.
                  - **activation='relu':** Função de ativação que introduz não-linearidade.
                  - **kernel_regularizer=regularizers.l2(0.001):** Aplicação da regularização L2 com taxa 0.001.

                **6. Camada de Saída (Dense)**
                - **O que é?**
                  Camada de saída que gera as probabilidades de cada classe usando a função de ativação softmax.
                - **Função:**
                  **Geração das Probabilidades:** Transforma as saídas das camadas densas em probabilidades para cada classe.
                - **Exemplo no Modelo:**
                  ```python
                  Dense(len(classes), activation='softmax')
                  ```
                  - **len(classes):** Número de classes a serem classificadas.
                  - **activation='softmax':** Função de ativação que transforma as saídas em probabilidades.
                """)

            # Definição dos Callbacks
            st.write("### Configurando Callbacks para o Treinamento...")
            st.write("""
            **Callbacks** são funções que são chamadas durante o treinamento da rede neural. Elas podem ser usadas para monitorar o desempenho do modelo e ajustar o treinamento de acordo com certos critérios. Nesta aplicação, utilizamos dois callbacks:
            - **ModelCheckpoint:** Salva o modelo automaticamente quando a métrica de validação melhora.
            - **EarlyStopping:** Interrompe o treinamento automaticamente se a métrica de validação não melhorar após um número especificado de épocas, evitando overfitting.
            """)

            diretorio_salvamento = 'modelos_salvos'
            if not os.path.exists(diretorio_salvamento):
                os.makedirs(diretorio_salvamento)
                st.write(f"**Diretório '{diretorio_salvamento}' criado para salvamento do modelo.**")
                logging.info(f"Diretório '{diretorio_salvamento}' criado.")
            else:
                st.write(f"**Diretório '{diretorio_salvamento}' já existe.**")
                logging.info(f"Diretório '{diretorio_salvamento}' já existe.")

            # Configuração do ModelCheckpoint
            checkpointer = ModelCheckpoint(
                filepath=os.path.join(diretorio_salvamento, 'modelo_agua_aumentado.keras'),  # Pode usar .h5 se preferir
                monitor='val_loss',
                verbose=1,
                save_best_only=True
            )

            # Parâmetros de EarlyStopping
            st.sidebar.subheader("Parâmetros de EarlyStopping")
            es_monitor = st.sidebar.selectbox(
                "Monitorar:",
                options=["val_loss", "val_accuracy"],
                index=0,
                help="Métrica a ser monitorada para EarlyStopping. 'val_loss' monitora a perda na validação, enquanto 'val_accuracy' monitora a acurácia na validação."
            )
            es_patience = st.sidebar.slider(
                "Paciência (Épocas):",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="Número de épocas sem melhoria antes de interromper o treinamento. Por exemplo, se 'patience' for 5, o treinamento será interrompido após 5 épocas sem melhoria na métrica monitorada."
            )
            es_mode = st.sidebar.selectbox(
                "Modo:",
                options=["min", "max"],
                index=0,
                help="Define se a métrica monitorada deve ser minimizada ('min') ou maximizada ('max'). 'val_loss' deve ser minimizada, enquanto 'val_accuracy' deve ser maximizada."
            )

            earlystop = EarlyStopping(
                monitor=es_monitor,
                patience=es_patience,
                restore_best_weights=True,
                mode=es_mode
            )

            # Definir as callbacks
            callbacks = [checkpointer, earlystop]

            # Treinamento do Modelo
            st.write("### Iniciando o Treinamento do Modelo...")
            st.write("""
            O treinamento pode demorar algum tempo, dependendo do tamanho do seu conjunto de dados e dos parâmetros selecionados. Durante o treinamento, as métricas de perda e acurácia serão exibidas para acompanhamento.
            """)
            with st.spinner('Treinando o modelo...'):
                if cross_validation and k_folds > 1:
                    # Implementar Validação Cruzada
                    st.write("**Validação Cruzada Iniciada**")
                    kf = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)
                    fold_no = 1
                    val_scores = []
                    for train_index, val_index in kf.split(X_train_combined):
                        st.write(f"#### Fold {fold_no}")
                        logging.info(f"Iniciando Fold {fold_no} de {k_folds}")
                        X_train_cv, X_val_cv = X_train_combined[train_index], X_train_combined[val_index]
                        y_train_cv, y_val_cv = y_train_combined[train_index], y_train_combined[val_index]

                        # Ajustar a forma dos dados
                        X_train_cv = X_train_cv.reshape((X_train_cv.shape[0], X_train_cv.shape[1], 1))
                        X_val_cv = X_val_cv.reshape((X_val_cv.shape[0], X_val_cv.shape[1], 1))

                        # Treinar o modelo
                        historico = modelo.fit(
                            X_train_cv, to_categorical(y_train_cv),
                            epochs=num_epochs,
                            batch_size=batch_size,
                            validation_data=(X_val_cv, to_categorical(y_val_cv)),
                            callbacks=callbacks,
                            class_weight=class_weight_dict,
                            verbose=1
                        )

                        # Avaliar no fold atual
                        score = modelo.evaluate(X_val_cv, to_categorical(y_val_cv), verbose=0)
                        st.write(f"**Acurácia no Fold {fold_no}:** {score[1]*100:.2f}%")
                        val_scores.append(score[1]*100)
                        logging.info(f"Fold {fold_no} Acurácia: {score[1]*100:.2f}%")
                        fold_no += 1

                    st.write(f"**Acurácia Média da Validação Cruzada ({k_folds}-Fold):** {np.mean(val_scores):.2f}%")
                    logging.info(f"Acurácia Média da Validação Cruzada: {np.mean(val_scores):.2f}%")
                else:
                    # Treinamento tradicional
                    historico = modelo.fit(
                        X_train_final, to_categorical(y_train_final),
                        epochs=num_epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, to_categorical(y_val)),
                        callbacks=callbacks,
                        class_weight=class_weight_dict,
                        verbose=1
                    )
                st.success("Treinamento concluído com sucesso!")
                logging.info("Treinamento concluído.")

            # Salvamento do Modelo e Classes
            st.write("### Download do Modelo Treinado e Arquivo de Classes")
            st.write("""
            Após o treinamento, você pode baixar o modelo treinado e o arquivo de classes para utilização futura ou para compartilhar com outros.
            """)

            # Salvar o modelo em um arquivo temporário com extensão .keras
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_model:
                modelo.save(tmp_model.name)
                caminho_tmp_model = tmp_model.name
                logging.info(f"Modelo salvo temporariamente em {caminho_tmp_model}.")

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
            logging.info(f"Arquivo temporário do modelo {caminho_tmp_model} removido.")

            # Salvar as classes
            classes_str = "\n".join(classes)
            st.download_button(
                label="Download das Classes (classes.txt)",
                data=classes_str,
                file_name="classes.txt",
                mime="text/plain"
            )
            logging.info("Arquivo de classes disponível para download.")

            # Avaliação do Modelo
            if not cross_validation:
                st.write("### Avaliação do Modelo nos Conjuntos de Treino, Validação e Teste")
                st.write("""
                A seguir, apresentamos a **Acurácia** do modelo nos conjuntos de treino, validação e teste. A acurácia representa a porcentagem de previsões corretas realizadas pelo modelo.
                """)
                score_train = modelo.evaluate(X_train_final, to_categorical(y_train_final), verbose=0)
                score_val = modelo.evaluate(X_val, to_categorical(y_val), verbose=0)
                score_test = modelo.evaluate(X_test, to_categorical(y_test), verbose=0)

                st.write(f"**Acurácia no Treino:** {score_train[1]*100:.2f}%")
                st.write(f"**Acurácia na Validação:** {score_val[1]*100:.2f}%")
                st.write(f"**Acurácia no Teste:** {score_test[1]*100:.2f}%")
                logging.info(f"Acurácia: Treino={score_train[1]*100:.2f}%, Validação={score_val[1]*100:.2f}%, Teste={score_test[1]*100:.2f}%")

                # **Explicação da Avaliação**
                with st.expander("📖 Explicação da Avaliação do Modelo"):
                    st.markdown("""
                    **Conclusão**

                    Entender os dados e as camadas do modelo é fundamental para interpretar como o modelo está aprendendo e realizando as classificações. 

                    - **Shapes dos Dados:**
                      - Representam a estrutura dos dados em diferentes etapas do processamento e treinamento.
                      - Ajustar corretamente as dimensões é crucial para que o modelo possa processar os dados de forma eficiente.

                    - **Camadas do Modelo:**
                      - Cada camada tem uma função específica que contribui para a extração e processamento das informações necessárias para a classificação.
                      - **Conv1D** detecta padrões, **Dropout** previne overfitting, **MaxPooling1D** reduz a dimensionalidade, **Flatten** prepara os dados para a camada densa, e **Dense** realiza a classificação final.

                    Compreender esses conceitos permite ajustar e otimizar o modelo de forma mais eficaz, melhorando sua performance e capacidade de generalização.
                    """)

                # Predições no Conjunto de Teste
                st.write("### Métricas Avançadas de Avaliação")
                st.write("""
                A seguir, apresentamos métricas avançadas como Curva ROC, Curva Precision-Recall e AUC para uma análise mais detalhada do desempenho do modelo.
                """)
                y_pred = modelo.predict(X_test)
                y_pred_classes = y_pred.argmax(axis=1)
                y_true = y_test  # y_test já está em formato inteiro

                # Matriz de Confusão com Seaborn
                st.write("""
                ### Matriz de Confusão
                A **Matriz de Confusão** mostra como as previsões do modelo se comparam com os rótulos reais. Cada célula representa o número de previsões para cada combinação de classe real e prevista.
                """)
                cm = confusion_matrix(y_true, y_pred_classes, labels=range(len(classes)))
                cm_df = pd.DataFrame(cm, index=classes, columns=classes)
                fig_cm, ax_cm = plt.subplots(figsize=(12,8))
                sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_title("Matriz de Confusão", fontsize=16)
                ax_cm.set_xlabel("Classe Prevista", fontsize=14)
                ax_cm.set_ylabel("Classe Real", fontsize=14)
                ax_cm.tick_params(axis='both', which='major', labelsize=12)
                st.pyplot(fig_cm)
                plt.close(fig_cm)
                logging.info("Matriz de Confusão exibida.")

                # Relatório de Classificação com Seaborn
                st.write("""
                ### Relatório de Classificação
                O **Relatório de Classificação** fornece métricas detalhadas sobre o desempenho do modelo em cada classe, incluindo precisão, recall e F1-score.
                """)
                report = classification_report(y_true, y_pred_classes, labels=range(len(classes)),
                                               target_names=classes, zero_division=0, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
                logging.info("Relatório de Classificação exibido.")

                # Curva ROC
                st.write("### Curva ROC")
                plot_roc_curve(y_true, y_pred, classes)

                # Curva Precision-Recall
                st.write("### Curva Precision-Recall")
                plot_precision_recall_curve_custom(y_true, y_pred, classes)

                # Visualizações das Métricas de Treinamento com Seaborn
                st.write("""
                ### Visualizações das Métricas de Treinamento
                As seguintes figuras mostram como a **Perda (Loss)** e a **Acurácia** evoluíram durante o treinamento e validação. Isso ajuda a entender como o modelo está aprendendo ao longo das épocas.
                """)
                historico_df = pd.DataFrame(historico.history)
                fig_loss, ax_loss = plt.subplots(figsize=(10,6))
                sns.lineplot(data=historico_df[['loss', 'val_loss']], ax=ax_loss)
                ax_loss.set_title("Perda (Loss) durante o Treinamento", fontsize=16)
                ax_loss.set_xlabel("Época", fontsize=14)
                ax_loss.set_ylabel("Loss", fontsize=14)
                ax_loss.tick_params(axis='both', which='major', labelsize=12)
                st.pyplot(fig_loss)
                plt.close(fig_loss)

                fig_acc, ax_acc = plt.subplots(figsize=(10,6))
                sns.lineplot(data=historico_df[['accuracy', 'val_accuracy']], ax=ax_acc)
                ax_acc.set_title("Acurácia durante o Treinamento", fontsize=16)
                ax_acc.set_xlabel("Época", fontsize=14)
                ax_acc.set_ylabel("Acurácia", fontsize=14)
                ax_acc.tick_params(axis='both', which='major', labelsize=12)
                st.pyplot(fig_acc)
                plt.close(fig_acc)
                logging.info("Curvas de Loss e Acurácia exibidas.")

                # Limpeza de Memória
                del modelo, historico, historico_df
                gc.collect()
                logging.info("Memória limpa após avaliação.")

                st.success("Processo de Treinamento e Avaliação concluído!")
            else:
                # Avaliação durante Cross-Validation (não exibido aqui para simplicidade)
                st.write("**Validação Cruzada concluída.**")
                logging.info("Validação Cruzada concluída.")

            # Visualizações de Experiment Tracking
            st.write("### Logs de Experimentos")
            st.write("""
            Acompanhe os detalhes dos experimentos realizados no arquivo `experiment_logs.log`. Isso inclui informações sobre configurações de treinamento, desempenho do modelo e quaisquer erros que possam ter ocorrido.
            """)
            try:
                with open('experiment_logs.log', 'r') as log_file:
                    logs = log_file.read()
                    st.text_area("Logs de Experimentos", logs, height=300)
            except Exception as e:
                st.error(f"Erro ao carregar logs: {e}")
                logging.error(f"Erro ao carregar logs: {e}")

            # Limpeza de Memória e Remoção dos Arquivos Temporários
            st.write("### Limpeza de Memória e Remoção de Arquivos Temporários")
            del df, X, y_valid, X_train, X_temp, y_train, y_temp, X_val, X_test, y_val, y_test
            if enable_augmentation:
                del X_train_augmented, y_train_augmented, X_train_combined, y_train_combined
            gc.collect()
            os.remove(caminho_zip)
            for cat in categorias:
                caminho_cat = os.path.join(caminho_base, cat)
                for arquivo in os.listdir(caminho_cat):
                    os.remove(os.path.join(caminho_cat, arquivo))
                os.rmdir(caminho_cat)
            os.rmdir(caminho_base)
            logging.info("Arquivos temporários removidos e memória limpa.")
            st.success("Processo de Treinamento e Avaliação concluído!")
        except Exception as e:
            st.error(f"Erro durante o processamento do dataset: {e}")
            logging.error(f"Erro durante o processamento do dataset: {e}")
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
            logging.info("Arquivos temporários removidos devido a erro.")

if __name__ == "__main__":
    # Chamada da função principal
    if app_mode == "Classificar Áudio":
        classificar_audio(SEED)
    elif app_mode == "Treinar Modelo":
        treinar_modelo(SEED)

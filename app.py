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
from PIL import Image, UnidentifiedImageError
import io
import torch
import zipfile
import gc

# ==================== CONFIGURAÇÃO DA PÁGINA ====================
def main():
    # ==================== DEFINIÇÃO DO SEED ====================
    st.sidebar.header("Configurações Gerais")
    
    # Adicionando a seleção de SEED na barra lateral
    seed_options = [0, 42, 100]
    seed_selection = st.sidebar.selectbox(
        "Escolha o valor do SEED:",
        options=seed_options,
        index=1,  # 42 como valor padrão
        help="Define a semente para reprodutibilidade dos resultados."
    )
    
    SEED = seed_selection  # Definindo a variável SEED

    # Definir o caminho do ícone (favicon)
    favicon_path = "logo.png"  # Verifique se o arquivo logo.png está no diretório correto

    # Verificar se o arquivo de ícone existe antes de configurá-lo
    if os.path.exists(favicon_path):
        try:
            st.set_page_config(
                page_title="Classificação de Sons de Água Vibrando em Copo de Vidro",
                page_icon=favicon_path,
                layout="wide"
            )
            # logging.info(f"Ícone {favicon_path} carregado com sucesso.")  # Remova ou configure o logging se necessário
        except Exception as e:
            st.set_page_config(
                page_title="Classificação de Sons de Água Vibrando em Copo de Vidro",
                layout="wide"
            )
            st.sidebar.warning(f"Erro ao carregar o ícone {favicon_path}: {e}")
    else:
        # Se o ícone não for encontrado, carrega sem favicon e exibe aviso
        st.set_page_config(
            page_title="Classificação de Sons de Água Vibrando em Copo de Vidro",
            layout="wide"
        )
        st.sidebar.warning(f"Ícone '{favicon_path}' não encontrado, carregando sem favicon.")

    # ==================== LOGO E IMAGEM DE CAPA ====================
    # Carrega e exibe a capa.png na página principal
    if os.path.exists('capa (2).png'):
        try:
            st.image(
                'capa (2).png', 
                caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', 
                use_container_width=True
            )
        except UnidentifiedImageError:
            st.warning("Imagem 'capa (2).png' não pôde ser carregada ou está corrompida.")
    else:
        st.warning("Imagem 'capa (2).png' não encontrada.")

    # Carregar o logotipo na barra lateral
    if os.path.exists("logo.png"):
        try:
            st.sidebar.image("logo.png", width=200, use_container_width=False)
        except UnidentifiedImageError:
            st.sidebar.text("Imagem do logotipo não pôde ser carregada ou está corrompida.")
    else:
        st.sidebar.text("Imagem do logotipo não encontrada.")

    st.title("Classificação de Sons de Água Vibrando em Copo de Vidro com Aumento de Dados e CNN")
    st.write("""
    Bem-vindo à nossa aplicação! Aqui, você pode **classificar sons de água vibrando em copos de vidro**. Você tem duas opções:
    - **Classificar Áudio:** Use um modelo já treinado para identificar o som.
    - **Treinar Modelo:** Treine seu próprio modelo com seus dados de áudio.
    """)

    # Barra Lateral de Navegação
    st.sidebar.title("Navegação")
    app_mode = st.sidebar.selectbox("Escolha a seção", ["Classificar Áudio", "Treinar Modelo"])

    if app_mode == "Classificar Áudio":
        classificar_audio(SEED)
    elif app_mode == "Treinar Modelo":
        treinar_modelo(SEED)
    
    # Adicionando o ícone na barra lateral
    if os.path.exists("eu.ico"):
        try:
            st.sidebar.image("eu.ico", width=80, use_container_width=False)
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

    Whatsapp: (88)981587145

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
    ax.set_ylabel("Frequência (Mel)", fontsize=14)
    
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
    
    # Correção do aviso de depreciação do Seaborn
    # Adicionando 'hue' e removendo a legenda
    sns.barplot(x=classes, y=probs, hue=classes, palette='viridis', ax=ax, legend=False)
    
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
        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {e}")
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

            # **Explicação dos Dados**
            with st.expander("📖 Explicação dos Dados"):
                st.markdown("""
                ### Explicação dos Dados

                **1. Features Extraídas: (10, 40)**
                - **O que são Features?**
                  Features são características ou informações específicas extraídas dos dados brutos (neste caso, arquivos de áudio) que são usadas para treinar o modelo.
                - **Interpretação de (10, 40):**
                  - **10:** Número de amostras ou exemplos no conjunto de dados.
                  - **40:** Número de características extraídas de cada amostra.
                - **Explicação Simples:**
                  Imagine que você tem 10 arquivos de áudio diferentes. Para cada um deles, extraímos 40 características que ajudam o modelo a entender e diferenciar os sons.

                **2. Divisão dos Dados:**
                Após extrair as features, os dados são divididos em diferentes conjuntos para treinar e avaliar o modelo.

                - **Treino: (8, 40)**
                  - **8:** Número de amostras usadas para treinar o modelo.
                  - **40:** Número de características por amostra.
                  - **Explicação:** Das 10 amostras iniciais, 8 são usadas para ensinar o modelo a reconhecer os padrões.

                - **Teste: (2, 40)**
                  - **2:** Número de amostras usadas para testar a performance do modelo.
                  - **40:** Número de características por amostra.
                  - **Explicação:** As 2 amostras restantes são usadas para verificar se o modelo aprendeu corretamente.

                **Dados Aumentados: (80, 40)**
                - **80:** Número de amostras adicionais geradas através de técnicas de aumento de dados.
                - **40:** Número de características por amostra.
                - **Explicação:** Para melhorar a performance do modelo, criamos 80 novas amostras a partir das originais, aplicando transformações como adicionar ruído ou alterar o pitch.
                """)

            # **Exibir Número de Classes e Distribuição**
            st.write(f"### Número de Classes: {len(classes)}")
            contagem_classes = df['classe'].value_counts()
            st.write("### Distribuição das Classes:")
            fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
            # Correção do aviso do Seaborn adicionando 'hue' e removendo a legenda
            sns.barplot(x=contagem_classes.index, y=contagem_classes.values, hue=contagem_classes.index, palette='viridis', ax=ax_dist, legend=False)
            ax_dist.set_xlabel("Classes", fontsize=14)
            ax_dist.set_ylabel("Número de Amostras", fontsize=14)
            ax_dist.set_title("Distribuição das Classes no Dataset", fontsize=16)
            ax_dist.tick_params(axis='both', which='major', labelsize=12)
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
                else:
                    st.warning(f"Erro no carregamento do arquivo '{arquivo}'.")

            X = np.array(X)
            y_valid = np.array(y_valid)

            st.write(f"**Features extraídas:** {X.shape}")

            # **Explicação das Features Extraídas**
            with st.expander("📖 Explicação das Features Extraídas"):
                st.markdown("""
                **1. Features Extraídas: (10, 40)**
                - **O que são Features?**
                  Features são características ou informações específicas extraídas dos dados brutos (neste caso, arquivos de áudio) que são usadas para treinar o modelo.
                - **Interpretação de (10, 40):**
                  - **10:** Número de amostras ou exemplos no conjunto de dados.
                  - **40:** Número de características extraídas de cada amostra.
                - **Explicação Simples:**
                  Imagine que você tem 10 arquivos de áudio diferentes. Para cada um deles, extraímos 40 características que ajudam o modelo a entender e diferenciar os sons.
                """)

            # Divisão dos Dados
            st.write("### Dividindo os Dados em Treino e Teste...")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_valid, test_size=0.2, random_state=SEED, stratify=y_valid)
            st.write(f"**Treino:** {X_train.shape}, **Teste:** {X_test.shape}")

            # **Explicação da Divisão dos Dados**
            with st.expander("📖 Explicação da Divisão dos Dados"):
                st.markdown("""
                **2. Divisão dos Dados:**
                Após extrair as features, os dados são divididos em diferentes conjuntos para treinar e avaliar o modelo.

                - **Treino: (8, 40)**
                  - **8:** Número de amostras usadas para treinar o modelo.
                  - **40:** Número de características por amostra.
                  - **Explicação:** Das 10 amostras iniciais, 8 são usadas para ensinar o modelo a reconhecer os padrões.

                - **Teste: (2, 40)**
                  - **2:** Número de amostras usadas para testar a performance do modelo.
                  - **40:** Número de características por amostra.
                  - **Explicação:** As 2 amostras restantes são usadas para verificar se o modelo aprendeu corretamente.
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
                    else:
                        st.warning(f"Erro no carregamento do arquivo '{arquivo}' para Data Augmentation.")

                X_train_augmented = np.array(X_train_augmented)
                y_train_augmented = np.array(y_train_augmented)
                st.write(f"**Dados aumentados:** {X_train_augmented.shape}")

                # **Explicação dos Dados Aumentados**
                with st.expander("📖 Explicação dos Dados Aumentados"):
                    st.markdown("""
                    **Dados Aumentados: (80, 40)**
                    - **80:** Número de amostras adicionais geradas através de técnicas de aumento de dados.
                    - **40:** Número de características por amostra.
                    - **Explicação:** Para melhorar a performance do modelo, criamos 80 novas amostras a partir das originais, aplicando transformações como adicionar ruído ou alterar o pitch.
                    """)

                # Combinação dos Dados
                X_train_combined = np.concatenate((X_train, X_train_augmented), axis=0)
                y_train_combined = np.concatenate((y_train, y_train_augmented), axis=0)
                st.write(f"**Treino combinado:** {X_train_combined.shape}")

                # **Explicação da Combinação dos Dados**
                with st.expander("📖 Explicação da Combinação dos Dados"):
                    st.markdown("""
                    **3. Combinação e Validação:**
                    - **Treino Combinado: (88, 40)**
                      - **88:** Soma das amostras de treino original (8) e aumentadas (80).
                      - **40:** Número de características por amostra.
                      - **Explicação:** Unimos as amostras originais com as aumentadas para formar um conjunto de treino mais robusto.
                    """)

            else:
                X_train_combined = X_train
                y_train_combined = y_train

            # Divisão em Treino Final e Validação
            st.write("### Dividindo o Treino Combinado em Treino Final e Validação...")
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train_combined, y_train_combined, test_size=0.1, random_state=SEED, stratify=y_train_combined)
            st.write(f"**Treino Final:** {X_train_final.shape}, **Validação:** {X_val.shape}")

            # **Explicação da Divisão Final**
            with st.expander("📖 Explicação da Combinação e Validação"):
                st.markdown("""
                **3. Combinação e Validação:**
                - **Treino Combinado: (88, 40)**
                  - **88:** Soma das amostras de treino original (8) e aumentadas (80).
                  - **40:** Número de características por amostra.
                  - **Explicação:** Unimos as amostras originais com as aumentadas para formar um conjunto de treino mais robusto.
                - **Treino Final: (79, 40)**
                  - **79:** Número de amostras após uma divisão adicional para validação.
                  - **40:** Número de características por amostra.
                  - **Explicação:** Das 88 amostras combinadas, 79 são usadas para treinar o modelo definitivamente.
                - **Validação: (9, 40)**
                  - **9:** Número de amostras usadas para validar o modelo durante o treinamento.
                  - **40:** Número de características por amostra.
                  - **Explicação:** As 9 amostras restantes são usadas para monitorar se o modelo está aprendendo de forma adequada.
                """)

            # Ajuste da Forma dos Dados para a CNN (Conv1D)
            st.write("### Ajustando a Forma dos Dados para a CNN (Conv1D)...")
            X_train_final = X_train_final.reshape((X_train_final.shape[0], X_train_final.shape[1], 1))
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            st.write(f"**Shapes:** Treino Final: {X_train_final.shape}, Validação: {X_val.shape}, Teste: {X_test.shape}")

            # **Explicação do Ajuste das Shapes**
            with st.expander("📖 Explicação do Ajuste das Shapes"):
                st.markdown("""
                **4. Ajuste das Shapes para a CNN:**
                Após a preparação dos dados, é necessário ajustar a shape (dimensões) dos dados para que sejam compatíveis com a Rede Neural Convolucional (CNN).

                - **Treino Final: (79, 40, 1)**
                - **Validação: (9, 40, 1)**
                - **Teste: (2, 40, 1)**
                
                - **Interpretação:**
                  - **79, 9, 2:** Número de amostras nos conjuntos de treino final, validação e teste, respectivamente.
                  - **40:** Número de características (features) por amostra.
                  - **1:** Número de canais. Neste caso, temos um único canal, pois estamos lidando com dados unidimensionais (áudio).

                - **Explicação Simples:**
                  Cada amostra de áudio agora tem uma dimensão extra (1) para indicar que há apenas um canal de informação, o que é necessário para processar os dados na CNN.
                """)

            # Cálculo de Class Weights
            st.write("### Calculando Class Weights para Balanceamento das Classes...")
            st.write("""
            **Class Weights** são utilizados para lidar com desequilíbrios nas classes do conjunto de dados. Quando algumas classes têm muito mais amostras do que outras, o modelo pode se tornar tendencioso em favor das classes mais frequentes. Aplicar pesos balanceados ajuda o modelo a prestar mais atenção às classes menos representadas.
            """)
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
            st.write("""
            A **Rede Neural Convolucional (CNN)** é uma arquitetura de rede neural eficaz para processamento de dados com estrutura de grade, como imagens e sinais de áudio. Nesta aplicação, utilizamos camadas convolucionais para extrair características relevantes dos dados de áudio.
            """)

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
                - **Exemplo no Modelo:**
                  ```python
                  Conv1D(64, kernel_size=10, activation='relu')
                  ```
                  - **64:** Número de filtros (detetores de padrões) usados.
                  - **kernel_size=10:** Tamanho da janela que percorre os dados.
                  - **activation='relu':** Função de ativação que introduz não-linearidade.

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

                **4. Conv1D_1 (Conv1D)**
                - **O que é?**
                  Outra camada convolucional para extrair padrões mais complexos dos dados.
                - **Função:**
                  Similar à primeira camada Conv1D, mas com mais filtros para capturar padrões mais elaborados.
                - **Exemplo no Modelo:**
                  ```python
                  Conv1D(128, kernel_size=10, activation='relu', padding='same')
                  ```
                  - **128:** Número de filtros.
                  - **kernel_size=10:** Tamanho da janela.
                  - **padding='same':** Mantém as dimensões dos dados.

                **5. Dropout_1 (Dropout)**
                - **O que é?**
                  Segunda camada de dropout para reforçar a regularização.
                - **Função:**
                  Similar à primeira camada Dropout.
                - **Exemplo no Modelo:**
                  ```python
                  Dropout(0.4)
                  ```
                  - **0.4:** 40% dos neurônios serão desligados.

                **6. MaxPooling1D_1 (MaxPooling1D)**
                - **O que é?**
                  Segunda camada de max pooling para continuar a reduzir a dimensionalidade.
                - **Função:**
                  Similar à primeira camada MaxPooling1D.
                - **Exemplo no Modelo:**
                  ```python
                  MaxPooling1D(pool_size=4)
                  ```
                  - **pool_size=4:** Seleciona o maior valor em janelas de 4 unidades.

                **7. Flatten (Flatten)**
                - **O que é?**
                  Flatten é uma camada que transforma os dados multidimensionais em um vetor unidimensional.
                - **Função:**
                  **Preparar para Camadas Densas:** Converte a saída das camadas convolucionais em uma forma adequada para as camadas densas (totalmente conectadas).
                - **Exemplo no Modelo:**
                  ```python
                  Flatten()
                  ```
                  - Sem parâmetros, apenas altera a forma dos dados.

                **8. Dense (Dense)**
                - **O que é?**
                  Dense é uma camada totalmente conectada onde cada neurônio está conectado a todos os neurônios da camada anterior.
                - **Função:**
                  **Tomar Decisões Finais:** Combina todas as características extraídas pelas camadas anteriores para fazer a classificação final.
                - **Exemplo no Modelo:**
                  ```python
                  Dense(64, activation='relu')
                  ```
                  - **64:** Número de neurônios na camada.
                  - **activation='relu':** Função de ativação que introduz não-linearidade.

                **9. Dropout_2 (Dropout)**
                - **O que é?**
                  Terceira camada de dropout para prevenir overfitting.
                - **Função:**
                  Similar às camadas Dropout anteriores.
                - **Exemplo no Modelo:**
                  ```python
                  Dropout(0.4)
                  ```
                  - **0.4:** 40% dos neurônios serão desligados.

                **10. Dense_1 (Dense)**
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

            from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
            diretorio_salvamento = 'modelos_salvos'
            if not os.path.exists(diretorio_salvamento):
                os.makedirs(diretorio_salvamento)
                st.write(f"**Diretório '{diretorio_salvamento}' criado para salvamento do modelo.**")
            else:
                st.write(f"**Diretório '{diretorio_salvamento}' já existe.**")

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

            # Treinamento do Modelo
            st.write("### Iniciando o Treinamento do Modelo...")
            st.write("""
            O treinamento pode demorar algum tempo, dependendo do tamanho do seu conjunto de dados e dos parâmetros selecionados. Durante o treinamento, as métricas de perda e acurácia serão exibidas para acompanhamento.
            """)
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
            st.write("""
            Após o treinamento, você pode baixar o modelo treinado e o arquivo de classes para utilização futura ou para compartilhar com outros.
            """)

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
            st.write("""
            A seguir, apresentamos a **Acurácia** do modelo nos conjuntos de treino, validação e teste. A acurácia representa a porcentagem de previsões corretas realizadas pelo modelo.
            """)
            score_train = modelo.evaluate(X_train_final, to_categorical(y_train_final), verbose=0)
            score_val = modelo.evaluate(X_val, to_categorical(y_val), verbose=0)
            score_test = modelo.evaluate(X_test, to_categorical(y_test), verbose=0)

            st.write(f"**Acurácia no Treino:** {score_train[1]*100:.2f}%")
            st.write(f"**Acurácia na Validação:** {score_val[1]*100:.2f}%")
            st.write(f"**Acurácia no Teste:** {score_test[1]*100:.2f}%")

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

            # Relatório de Classificação com Seaborn
            st.write("""
            ### Relatório de Classificação
            O **Relatório de Classificação** fornece métricas detalhadas sobre o desempenho do modelo em cada classe, incluindo precisão, recall e F1-score.
            """)
            report = classification_report(y_true, y_pred_classes, labels=range(len(classes)),
                                           target_names=classes, zero_division=0, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

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

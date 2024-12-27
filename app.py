# Importação de módulos necessários para a aplicação:
# Esses módulos são usados para manipulação de arquivos, cálculos, visualizações, aprendizado de máquina e processamento de áudio.

import os  # Módulo para interagir com o sistema operacional, como manipulação de arquivos e diretórios.
import zipfile  # Permite lidar com arquivos ZIP para compactação e descompactação.
import shutil  # Fornece operações de nível superior com arquivos e coleções de arquivos.
import tempfile  # Criação de arquivos e diretórios temporários.
import random  # Ferramentas para gerar números aleatórios, útil para experimentos e particionamento aleatório.
import numpy as np  # Biblioteca para operações matemáticas e manipulação eficiente de arrays multidimensionais.
import pandas as pd  # Usado para manipular dados tabulares em DataFrames (dados estruturados em formato de tabela).
import matplotlib.pyplot as plt  # Biblioteca para criar gráficos 2D e visualizações de dados.
import seaborn as sns  # Extensão do Matplotlib para criar visualizações estatísticas atraentes e informativas.

from PIL import Image  # Biblioteca para abrir, manipular e salvar imagens em diversos formatos.
import torch  # Framework para aprendizado de máquina baseado em PyTorch, com suporte a tensores e redes neurais.
from torch import nn, optim  # Módulos para criar redes neurais (nn) e definir otimizadores (optim).
from torch.utils.data import DataLoader, TensorDataset  # Utilitários para carregar dados e transformá-los em tensores para treinamento.

from sklearn.model_selection import StratifiedKFold  # Particionamento estratificado para validação cruzada, garantindo que a proporção de classes seja mantida em cada partição.
from sklearn.metrics import (  # Métricas usadas para avaliar o desempenho do modelo.
    classification_report,  # Relatório de métricas por classe, como precisão, recall e F1-score.
    confusion_matrix,  # Matriz de confusão para medir erros entre classes.
    roc_curve,  # Função para calcular a Curva ROC (Receiver Operating Characteristic).
    auc,  # Área sob a curva ROC.
    f1_score,  # Métrica combinada de precisão e recall.
    mean_squared_error  # Erro quadrático médio, usado em problemas de regressão.
)

from imblearn.over_sampling import SMOTE  # Técnica de oversampling que gera novas amostras sintéticas para classes minoritárias.
from imblearn.under_sampling import RandomUnderSampler  # Técnica para reduzir o tamanho da classe majoritária.

import streamlit as st  # Biblioteca para criar interfaces interativas de usuário em Python para análise de dados e aprendizado de máquina.

import gc  # Coletor de lixo, usado para liberar memória não utilizada.
import logging  # Módulo para registrar mensagens e informações durante a execução do programa.
import io  # Manipulação de fluxos de entrada/saída, como arquivos em memória.

# TensorFlow e bibliotecas relacionadas para aprendizado profundo e processamento de áudio:
import tensorflow as tf  # Framework para aprendizado de máquina baseado em tensores.
import tensorflow_hub as hub  # Repositório de módulos prontos do TensorFlow, como modelos pré-treinados.

from scipy.io import wavfile  # Para ler e escrever arquivos de áudio em formato WAV.
import scipy.signal  # Ferramentas de processamento de sinais, como filtros e transformadas de Fourier.

import librosa  # Biblioteca para processamento de áudio, especializada em análise de frequências e características musicais.
import librosa.display  # Ferramentas para exibir dados de áudio, como espectrogramas.

import requests  # Para baixar arquivos e acessar URLs, como downloads de arquivos de áudio.

# Operações matemáticas e estatísticas:
import math  # Operações matemáticas básicas, como logaritmos e potências.
import statistics  # Ferramentas estatísticas simples, como médias e desvios padrão.

# Music21: Ferramenta para análise e manipulação de partituras musicais:
import music21

# Streamlit Components: Adição de componentes personalizados em Streamlit:
import streamlit.components.v1 as components

# Pretty MIDI: Biblioteca para manipulação de arquivos MIDI:
import pretty_midi

# SoundFile: Para ler e escrever arquivos de áudio em formatos como FLAC e WAV:
import soundfile as sf

# FluidSynth: Para converter arquivos MIDI em arquivos de áudio:
from midi2audio import FluidSynth

# Scikit-learn: Ferramentas para aprendizado de máquina supervisionado e pré-processamento:
from sklearn.decomposition import PCA  # Redução de dimensionalidade usando análise de componentes principais.
from sklearn.preprocessing import StandardScaler, label_binarize  # Normalização de dados e binarização de rótulos.
from sklearn.linear_model import LogisticRegression  # Classificador de regressão logística.
from sklearn.ensemble import RandomForestClassifier  # Classificador baseado em árvores de decisão.
from sklearn.pipeline import Pipeline  # Facilita a combinação de etapas de pré-processamento e classificação.
from sklearn.feature_selection import SelectKBest, f_classif  # Seleção de melhores características com base em testes estatísticos.
from sklearn.neural_network import MLPClassifier  # Classificador baseado em redes neurais de múltiplas camadas.

# Suprimir avisos relacionados ao `torch.classes`
# Alguns avisos gerados pelo PyTorch podem ser irrelevantes para o contexto do projeto.
# Esta configuração ignora avisos específicos do tipo `UserWarning` contendo "torch.classes" na mensagem.
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.classes.*")

# Forçar o uso da CPU no TensorFlow e PyTorch
# Mesmo que uma GPU esteja disponível, o código é configurado para executar exclusivamente na CPU.
# Isso é útil para compatibilidade, evitar problemas de alocação de memória em GPUs, ou garantir resultados determinísticos.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Desativa GPUs para TensorFlow.
device = torch.device("cpu")  # Configura o PyTorch para usar a CPU explicitamente.

# Configurações para gráficos mais bonitos com o Seaborn
# O estilo 'whitegrid' adiciona um plano de fundo com uma grade clara, útil para visualizar os dados com maior clareza.
sns.set_style('whitegrid')

# Definir seed para reprodutibilidade
# A "seed" é usada para garantir que os resultados sejam consistentes entre execuções.
# Isso é fundamental em experimentos científicos para validar hipóteses.
# Esta função configura seeds para as bibliotecas Python, NumPy e PyTorch.
def set_seed(seed):
    random.seed(seed)  # Define a seed para a biblioteca `random`.
    np.random.seed(seed)  # Define a seed para o NumPy.
    torch.manual_seed(seed)  # Define a seed para operações do PyTorch.
    
    # Garantir que as operações em PyTorch sejam determinísticas.
    torch.backends.cudnn.deterministic = True  # Força resultados consistentes no uso de operações convolucionais.
    torch.backends.cudnn.benchmark = False  # Desativa otimizações dinâmicas que poderiam gerar resultados não determinísticos.

# Inicialização com seed fixa para reprodutibilidade
set_seed(42)  # A seed de valor 42 é amplamente usada por convenção.

# Definir constantes globalmente
# Lista com os nomes das 12 notas musicais usadas na notação padrão ocidental.
note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Frequência padrão do Lá central (A4) em Hz.
A4 = 440

# Calcular a frequência do Dó mais grave (C0) usando a relação harmônica:
# Frequência = A4 * 2^(-4.75), onde -4.75 corresponde à diferença em tons entre A4 e C0.
C0 = A4 * pow(2, -4.75)

# Função decorada para cachear o modelo YAMNet
@st.cache_resource
def load_yamnet_model():
    """
    Carrega o modelo YAMNet do TensorFlow Hub.
    
    O modelo YAMNet é projetado para tarefas de classificação de áudio, como identificar sons 
    do ambiente ou categorias de ruídos específicos. Ele é baseado em redes neurais profundas.
    
    O uso do decorador `@st.cache_resource` permite que o modelo carregado seja armazenado em cache.
    Assim, se a função for chamada novamente durante a execução do Streamlit, o modelo já carregado
    será reutilizado, economizando tempo e recursos computacionais.
    """
    yam_model = hub.load('https://tfhub.dev/google/yamnet/1')  # URL para o modelo YAMNet no TensorFlow Hub.
    return yam_model  # Retorna o modelo carregado.

# Decorador para cachear dados frequentemente utilizados
@st.cache_data
def load_class_map(url):
    """
    Carrega o mapa de classes do modelo YAMNet a partir de um arquivo CSV remoto.

    O YAMNet utiliza 521 categorias de sons, e o mapa de classes mapeia cada índice 
    de saída do modelo para uma categoria compreensível (como "latido de cachorro" ou "motor").

    Args:
        url (str): Endereço do arquivo CSV contendo o mapa de classes.

    Returns:
        pd.DataFrame: Um DataFrame contendo as categorias de áudio e seus respectivos índices.
    """
    class_map = pd.read_csv(url)  # Lê o arquivo CSV diretamente da URL fornecida.
    return class_map  # Retorna o DataFrame carregado.

def ensure_sample_rate(original_sr, waveform, desired_sr=16000):
    """
    Ajusta a taxa de amostragem de um áudio para o valor desejado (16 kHz por padrão).

    A taxa de amostragem (`sample rate`) define o número de amostras por segundo em um áudio.
    Muitos modelos de aprendizado de máquina de áudio (como o YAMNet) requerem uma taxa de 
    amostragem específica para funcionar corretamente.

    Args:
        original_sr (int): Taxa de amostragem original do áudio.
        waveform (np.ndarray): Sinal de áudio como uma sequência de valores.
        desired_sr (int): Taxa de amostragem desejada. Padrão: 16000 Hz.

    Returns:
        tuple: (int, np.ndarray) - Nova taxa de amostragem e o waveform ajustado.
    """
    if original_sr != desired_sr:  # Verifica se a taxa de amostragem precisa ser ajustada.
        # Calcula o novo tamanho do waveform após o ajuste.
        desired_length = int(round(float(len(waveform)) / original_sr * desired_sr))
        
        # Ajusta o waveform usando interpolação (resampling).
        waveform = scipy.signal.resample(waveform, desired_length)
    
    return desired_sr, waveform  # Retorna a nova taxa de amostragem e o waveform ajustado.

def add_noise(waveform, noise_factor=0.005):
    """
    Adiciona ruído branco ao áudio para data augmentation.

    Ruído branco é uma série de valores aleatórios que simulam sons ambientais ou interferências.
    Esse processo ajuda o modelo a se tornar mais robusto, ou seja, menos sensível a pequenas 
    alterações ou ruídos nos dados de entrada.

    Args:
        waveform (np.ndarray): O áudio original representado como uma sequência de valores numéricos.
        noise_factor (float): Intensidade do ruído a ser adicionado. Valores menores resultam em 
                              ruído mais sutil. Padrão: 0.005.

    Returns:
        np.ndarray: O áudio modificado com o ruído branco adicionado.
    """
    noise = np.random.randn(len(waveform))  # Gera ruído branco com o mesmo comprimento do áudio.
    augmented_waveform = waveform + noise_factor * noise  # Adiciona o ruído ao áudio original.
    augmented_waveform = augmented_waveform.astype(np.float32)  # Converte o áudio para float32.
    return augmented_waveform  # Retorna o áudio com ruído adicionado.

def time_stretch(waveform, rate=1.1):
    """
    Estica ou comprime o tempo do áudio sem alterar sua altura.

    Esse método altera a velocidade do áudio, tornando-o mais rápido ou mais lento, dependendo 
    do fator de esticamento (`rate`). É útil para simular variações de execução de áudio.

    Args:
        waveform (np.ndarray): O áudio original como uma sequência de valores numéricos.
        rate (float): Fator de esticamento do tempo. Valores > 1 aumentam a velocidade, enquanto
                      valores < 1 diminuem a velocidade. Padrão: 1.1.

    Returns:
        np.ndarray: O áudio com o tempo alterado.
    """
    return librosa.effects.time_stretch(waveform, rate)  # Utiliza a função integrada do Librosa.

def apply_pitch_shift(waveform, sr, n_steps=2):
    """
    Altera a altura (pitch) do áudio sem modificar sua duração.

    Modificar a altura significa alterar a frequência das ondas sonoras, resultando em tons mais
    agudos ou mais graves. Esse processo é realizado em semitons, onde cada "step" representa 
    meio tom na escala musical.

    Args:
        waveform (np.ndarray): O áudio original como uma sequência de valores numéricos.
        sr (int): Taxa de amostragem do áudio (sample rate), que é usada para calcular as frequências.
        n_steps (int): Número de semitons para alterar a altura. Valores positivos tornam o áudio 
                       mais agudo, enquanto valores negativos tornam-no mais grave. Padrão: 2.

    Returns:
        np.ndarray: O áudio com a altura alterada.
    """
    return librosa.effects.pitch_shift(waveform, sr, n_steps)  # Altera o pitch usando a função do Librosa.
def perform_data_augmentation(waveform, sr, augmentation_methods, rate=1.1, n_steps=2):
    """
    Aplica técnicas de data augmentation no áudio.

    O objetivo do data augmentation é aumentar a diversidade dos dados de treinamento
    gerando variações artificiais no áudio original. Isso ajuda o modelo a generalizar
    melhor e a ser mais robusto em diferentes cenários de entrada.

    Args:
        waveform (np.ndarray): O áudio original representado como uma sequência de valores numéricos.
        sr (int): Taxa de amostragem do áudio (sample rate), usada em operações como alteração de pitch.
        augmentation_methods (list): Lista de métodos de augmentação a serem aplicados. Pode incluir:
            - 'Adicionar Ruído': Adiciona ruído branco ao áudio.
            - 'Esticar Tempo': Modifica a velocidade do áudio sem alterar o pitch.
            - 'Mudar Pitch': Altera o tom do áudio sem alterar sua duração.
        rate (float): Fator de esticamento do tempo, utilizado apenas no método 'Esticar Tempo'.
                      Valores > 1 aumentam a velocidade; valores < 1 diminuem. Padrão: 1.1.
        n_steps (int): Número de semitons para alterar a altura (pitch), utilizado no método 'Mudar Pitch'.
                       Padrão: 2.

    Returns:
        list: Uma lista contendo o áudio original e as variações geradas pelas técnicas de augmentação.
    """
    # Inicializa a lista de áudios aumentados com o áudio original.
    augmented_waveforms = [waveform]

    # Itera sobre os métodos de augmentação especificados.
    for method in augmentation_methods:
        if method == 'Adicionar Ruído':
            # Aplica o método de adição de ruído branco.
            augmented_waveforms.append(add_noise(waveform))
        elif method == 'Esticar Tempo':
            try:
                # Aplica a esticamento de tempo.
                stretched = time_stretch(waveform, rate=rate)
                augmented_waveforms.append(stretched)
            except Exception as e:
                # Em caso de erro, exibe uma mensagem de aviso no Streamlit.
                st.warning(f"Erro ao aplicar Esticar Tempo: {e}")
        elif method == 'Mudar Pitch':
            try:
                # Aplica mudança de pitch.
                shifted = apply_pitch_shift(waveform, sr, n_steps=n_steps)
                augmented_waveforms.append(shifted)
            except Exception as e:
                # Em caso de erro, exibe uma mensagem de aviso no Streamlit.
                st.warning(f"Erro ao aplicar Mudar Pitch: {e}")

    # Retorna a lista de áudios aumentados.
    return augmented_waveforms
def extract_yamnet_embeddings(yamnet_model, audio_path):
    """
    Extrai embeddings usando o modelo YAMNet para um arquivo de áudio.
    Retorna a classe predita e a média dos embeddings das frames.

    O modelo YAMNet é uma rede neural projetada para analisar áudio e identificar
    categorias sonoras, como latidos, música ou ruídos de fundo. Ele divide o áudio
    em "frames" curtos e gera embeddings e pontuações para cada frame.

    Args:
        yamnet_model: Modelo YAMNet carregado do TensorFlow Hub.
        audio_path (str): Caminho para o arquivo de áudio a ser analisado.

    Returns:
        tuple: Classe predita (int) e vetor médio de embeddings (np.ndarray).
    """
    # Extrai apenas o nome do arquivo do caminho completo.
    basename_audio = os.path.basename(audio_path)

    try:
        # Lê o arquivo de áudio usando `wavfile`.
        sr_orig, wav_data = wavfile.read(audio_path)
        st.write(f"Processando {basename_audio}: Taxa de Amostragem = {sr_orig} Hz, Forma = {wav_data.shape}, Tipo de Dados = {wav_data.dtype}")

        # Verifica se o áudio está em estéreo.
        if wav_data.ndim > 1:
            # Converte áudio estéreo para mono calculando a média dos canais.
            wav_data = wav_data.mean(axis=1)
            st.write(f"Convertido para mono: Forma = {wav_data.shape}")
        
        # Normaliza o áudio para o intervalo [-1, 1].
        if wav_data.dtype.kind == 'i':
            # Caso o áudio esteja em formato inteiro (por exemplo, `int16`).
            max_val = np.iinfo(wav_data.dtype).max  # Encontra o valor máximo possível do tipo.
            waveform = wav_data / max_val  # Divide pelo valor máximo para normalizar.
            st.write(f"Normalizado de inteiros para float: max_val = {max_val}")
        elif wav_data.dtype.kind == 'f':
            # Caso o áudio já esteja em formato de ponto flutuante (por exemplo, `float32`).
            waveform = wav_data
            if np.max(waveform) > 1.0 or np.min(waveform) < -1.0:
                # Reescala se estiver fora do intervalo [-1.0, 1.0].
                waveform = waveform / np.max(np.abs(waveform))
                st.write("Normalizado para o intervalo [-1.0, 1.0]")
        else:
            # Tipo de dado não suportado.
            raise ValueError(f"Tipo de dado do áudio não suportado: {wav_data.dtype}")
    
        # Garante que o áudio está no formato `float32`.
        waveform = waveform.astype(np.float32)
    
        # Ajusta a taxa de amostragem para 16 kHz, conforme exigido pelo YAMNet.
        sr, waveform = ensure_sample_rate(sr_orig, waveform)
        st.write(f"Taxa de Amostragem ajustada: {sr} Hz")
    
        # Executa o modelo YAMNet no waveform processado.
        # A saída do modelo inclui:
        # - scores: Matriz de pontuações para cada frame e cada classe.
        # - embeddings: Representação vetorial de cada frame.
        # - spectrogram: Espectrograma processado pelo modelo.
        scores, embeddings, spectrogram = yamnet_model(waveform)
        st.write(f"Embeddings extraídos: Forma = {embeddings.shape}")
    
        # Extrai a matriz de pontuações e calcula a média das pontuações por frame.
        scores_np = scores.numpy()
        mean_scores = scores_np.mean(axis=0)  # Calcula a média ao longo dos frames.
        pred_class = mean_scores.argmax()  # Encontra o índice da classe com maior pontuação média.
        st.write(f"Classe predita pelo YAMNet: {pred_class}")
    
        # Calcula a média dos embeddings ao longo dos frames.
        # Isso cria uma única representação vetorial do áudio completo.
        mean_embedding = embeddings.numpy().mean(axis=0)  # Forma final: (1024,)
        st.write(f"Média dos embeddings das frames: Forma = {mean_embedding.shape}")
    
        return pred_class, mean_embedding  # Retorna a classe predita e os embeddings médios.
    except Exception as e:
        # Captura erros durante o processamento e exibe uma mensagem de erro no Streamlit.
        st.error(f"Erro ao processar {basename_audio}: {e}")
        return -1, None  # Retorna valores de fallback em caso de falha.

def extract_yamnet_embeddings(yamnet_model, audio_path):
    """
    Extrai embeddings usando o modelo YAMNet para um arquivo de áudio.
    Retorna a classe predita e a média dos embeddings das frames.

    O modelo YAMNet é uma rede neural projetada para analisar áudio e identificar
    categorias sonoras, como latidos, música ou ruídos de fundo. Ele divide o áudio
    em "frames" curtos e gera embeddings e pontuações para cada frame.

    Args:
        yamnet_model: Modelo YAMNet carregado do TensorFlow Hub.
        audio_path (str): Caminho para o arquivo de áudio a ser analisado.

    Returns:
        tuple: Classe predita (int) e vetor médio de embeddings (np.ndarray).
    """
    # Extrai apenas o nome do arquivo do caminho completo.
    basename_audio = os.path.basename(audio_path)

    try:
        # Lê o arquivo de áudio usando `wavfile`.
        sr_orig, wav_data = wavfile.read(audio_path)
        st.write(f"Processando {basename_audio}: Taxa de Amostragem = {sr_orig} Hz, Forma = {wav_data.shape}, Tipo de Dados = {wav_data.dtype}")

        # Verifica se o áudio está em estéreo.
        if wav_data.ndim > 1:
            # Converte áudio estéreo para mono calculando a média dos canais.
            wav_data = wav_data.mean(axis=1)
            st.write(f"Convertido para mono: Forma = {wav_data.shape}")
        
        # Normaliza o áudio para o intervalo [-1, 1].
        if wav_data.dtype.kind == 'i':
            # Caso o áudio esteja em formato inteiro (por exemplo, `int16`).
            max_val = np.iinfo(wav_data.dtype).max  # Encontra o valor máximo possível do tipo.
            waveform = wav_data / max_val  # Divide pelo valor máximo para normalizar.
            st.write(f"Normalizado de inteiros para float: max_val = {max_val}")
        elif wav_data.dtype.kind == 'f':
            # Caso o áudio já esteja em formato de ponto flutuante (por exemplo, `float32`).
            waveform = wav_data
            if np.max(waveform) > 1.0 or np.min(waveform) < -1.0:
                # Reescala se estiver fora do intervalo [-1.0, 1.0].
                waveform = waveform / np.max(np.abs(waveform))
                st.write("Normalizado para o intervalo [-1.0, 1.0]")
        else:
            # Tipo de dado não suportado.
            raise ValueError(f"Tipo de dado do áudio não suportado: {wav_data.dtype}")
    
        # Garante que o áudio está no formato `float32`.
        waveform = waveform.astype(np.float32)
    
        # Ajusta a taxa de amostragem para 16 kHz, conforme exigido pelo YAMNet.
        sr, waveform = ensure_sample_rate(sr_orig, waveform)
        st.write(f"Taxa de Amostragem ajustada: {sr} Hz")
    
        # Executa o modelo YAMNet no waveform processado.
        # A saída do modelo inclui:
        # - scores: Matriz de pontuações para cada frame e cada classe.
        # - embeddings: Representação vetorial de cada frame.
        # - spectrogram: Espectrograma processado pelo modelo.
        scores, embeddings, spectrogram = yamnet_model(waveform)
        st.write(f"Embeddings extraídos: Forma = {embeddings.shape}")
    
        # Extrai a matriz de pontuações e calcula a média das pontuações por frame.
        scores_np = scores.numpy()
        mean_scores = scores_np.mean(axis=0)  # Calcula a média ao longo dos frames.
        pred_class = mean_scores.argmax()  # Encontra o índice da classe com maior pontuação média.
        st.write(f"Classe predita pelo YAMNet: {pred_class}")
    
        # Calcula a média dos embeddings ao longo dos frames.
        # Isso cria uma única representação vetorial do áudio completo.
        mean_embedding = embeddings.numpy().mean(axis=0)  # Forma final: (1024,)
        st.write(f"Média dos embeddings das frames: Forma = {mean_embedding.shape}")
    
        return pred_class, mean_embedding  # Retorna a classe predita e os embeddings médios.
    except Exception as e:
        # Captura erros durante o processamento e exibe uma mensagem de erro no Streamlit.
        st.error(f"Erro ao processar {basename_audio}: {e}")
        return -1, None  # Retorna valores de fallback em caso de falha.
def balance_classes(X, y, method):
    """
    Balanceia as classes usando oversampling ou undersampling.

    Classes desbalanceadas ocorrem quando há mais exemplos de algumas categorias do que outras.
    Isso pode afetar negativamente o desempenho de modelos de aprendizado de máquina, que podem 
    se tornar enviesados para as classes majoritárias.

    Esta função implementa duas estratégias:
    1. Oversampling (aumentar o número de amostras das classes minoritárias usando SMOTE).
    2. Undersampling (reduzir o número de amostras das classes majoritárias usando RandomUnderSampler).

    Args:
        X (array-like): Matriz de características dos dados.
        y (array-like): Vetor de rótulos correspondente às classes.
        method (str): Método de balanceamento a ser aplicado. Pode ser 'Oversample', 'Undersample', ou 'Nenhum'.

    Returns:
        tuple: Dados balanceados (X_bal) e rótulos correspondentes (y_bal).
    """
    if method == 'Oversample':  # Se o método escolhido for oversampling
        smote = SMOTE(random_state=42)  # Synthetic Minority Oversampling Technique
        X_bal, y_bal = smote.fit_resample(X, y)  # Gera novas amostras para as classes minoritárias
    elif method == 'Undersample':  # Se o método escolhido for undersampling
        rus = RandomUnderSampler(random_state=42)  # Reduz o número de amostras das classes majoritárias
        X_bal, y_bal = rus.fit_resample(X, y)
    else:  # Se nenhum método for escolhido, retorna os dados originais
        X_bal, y_bal = X, y
    return X_bal, y_bal


def train_audio_classifier(X_train, y_train, X_val, y_val, input_dim, num_classes, classes, epochs, learning_rate, batch_size, l2_lambda, patience):
    """
    Treina um classificador avançado em PyTorch com os embeddings extraídos.
    Inclui validação cruzada e métricas de avaliação.
    
    Args:
        X_train, y_train: Dados de treino e seus rótulos.
        X_val, y_val: Dados de validação e seus rótulos.
        input_dim: Dimensão das entradas (número de características nos embeddings).
        num_classes: Número de classes no problema.
        classes: Lista de nomes das classes.
        epochs: Número de épocas para o treinamento.
        learning_rate: Taxa de aprendizado do otimizador.
        batch_size: Tamanho dos lotes para treinamento.
        l2_lambda: Fator de regularização L2 para evitar overfitting.
        patience: Número de épocas sem melhora antes de aplicar early stopping.

    Returns:
        O modelo treinado.
    """
    # **Preparação dos dados**: converte os conjuntos de treino e validação em tensores PyTorch.
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

    # Criar DataLoaders para manipular os dados em lotes durante o treinamento.
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle para aleatoriedade.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Sem shuffle para validação.

    # **Definição do modelo**: estrutura avançada com camadas totalmente conectadas, normalização e dropout.
    class AdvancedAudioClassifier(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(AdvancedAudioClassifier, self).__init__()
            self.fc1 = nn.Linear(input_dim, 512)  # Primeira camada com 512 unidades.
            self.bn1 = nn.BatchNorm1d(512)  # Normalização em lotes para estabilizar o treinamento.
            self.relu1 = nn.ReLU()  # Função de ativação ReLU.
            self.dropout1 = nn.Dropout(0.3)  # Dropout para evitar overfitting.
            self.fc2 = nn.Linear(512, 256)  # Segunda camada com 256 unidades.
            self.bn2 = nn.BatchNorm1d(256)  # Outra normalização em lotes.
            self.relu2 = nn.ReLU()  # Outra ReLU.
            self.dropout2 = nn.Dropout(0.3)  # Outro Dropout.
            self.fc3 = nn.Linear(256, num_classes)  # Camada final que mapeia para o número de classes.

        def forward(self, x):
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            return x

    classifier = AdvancedAudioClassifier(input_dim, num_classes).to(device)  # Instanciar o modelo na CPU ou GPU.

    # Definir função de perda (CrossEntropy para classificação) e otimizador Adam com regularização L2.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    # **Early stopping**: Parar o treinamento se não houver melhora por um número consecutivo de épocas.
    best_val_loss = float('inf')  # Melhor perda de validação observada.
    epochs_no_improve = 0  # Contador de épocas sem melhora.
    best_model_wts = None  # Armazenar os pesos do melhor modelo.

    # Inicializar listas para registrar métricas durante o treinamento.
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    progress_bar = st.progress(0)  # Barra de progresso para feedback visual.
    for epoch in range(epochs):
        classifier.train()  # Colocar o modelo em modo de treinamento.
        running_loss = 0.0
        running_corrects = 0

        # **Treinamento em lotes (batches)**.
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zerar os gradientes do otimizador.
            outputs = classifier(inputs)  # Fazer a previsão.
            _, preds = torch.max(outputs, 1)  # Predição final (classe com maior probabilidade).
            loss = criterion(outputs, labels)  # Calcular a perda.
            loss.backward()  # Backpropagation para calcular gradientes.
            optimizer.step()  # Atualizar os pesos.

            # Atualizar métricas acumuladas.
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # **Métricas por época no conjunto de treino.**
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        # **Avaliação no conjunto de validação.**
        classifier.eval()  # Colocar o modelo em modo de avaliação.
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():  # Desligar o cálculo de gradientes para economia de memória.
            for inputs, labels in val_loader:
                outputs = classifier(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc.item())

        # Atualizar a barra de progresso.
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)

        # Exibir métricas da época atual.
        st.write(f"### Época {epoch+1}/{epochs}")
        st.write(f"**Treino - Perda:** {epoch_loss:.4f}, **Acurácia:** {epoch_acc.item():.4f}")
        st.write(f"**Validação - Perda:** {val_epoch_loss:.4f}, **Acurácia:** {val_epoch_acc.item():.4f}")

        # Atualizar o modelo com early stopping.
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
            best_model_wts = classifier.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                st.write("Early stopping!")  # Interromper o treinamento se não houver melhora.
                break

    # Carregar os melhores pesos do modelo.
    if best_model_wts is not None:
        classifier.load_state_dict(best_model_wts)

    # Retorna o classificador treinado.
    return classifier
def midi_to_wav(midi_path, wav_path, soundfont_path):
    """
    Converte um arquivo MIDI para WAV usando FluidSynth via midi2audio ou, 
    caso não disponível, uma síntese básica de ondas senoidais.
    
    Args:
        midi_path (str): Caminho do arquivo MIDI de entrada.
        wav_path (str): Caminho para salvar o arquivo WAV gerado.
        soundfont_path (str): Caminho para o arquivo SoundFont usado pelo FluidSynth.

    Returns:
        bool: True se a conversão foi bem-sucedida, False caso contrário.
    """
    # Verifica se o FluidSynth está instalado no sistema.
    if is_fluidsynth_installed():
        try:
            fs = FluidSynth(soundfont_path)  # Inicializa o FluidSynth com o SoundFont fornecido.
            fs.midi_to_audio(midi_path, wav_path)  # Converte o arquivo MIDI em áudio WAV.
            return True
        except Exception as e:
            # Captura erros e tenta alternativa de síntese básica.
            st.error(f"Erro ao converter MIDI para WAV usando FluidSynth: {e}")
            st.warning("Tentando usar a síntese simples de ondas senoidais como alternativa.")
            return midi_to_wav_simple(midi_path, wav_path)
    else:
        # Caso o FluidSynth não esteja instalado, usa síntese básica.
        return midi_to_wav_simple(midi_path, wav_path)

def is_fluidsynth_installed():
    """
    Verifica se o FluidSynth está instalado no sistema e acessível no PATH.
    
    Returns:
        bool: True se o FluidSynth estiver instalado, False caso contrário.
    """
    return shutil.which("fluidsynth") is not None  # Verifica a existência do comando 'fluidsynth'.

def midi_to_wav_simple(midi_path, wav_path):
    """
    Converte um arquivo MIDI para WAV usando uma abordagem simples de síntese de ondas senoidais.
    Esta implementação é limitada e gera áudio básico.
    
    Args:
        midi_path (str): Caminho do arquivo MIDI de entrada.
        wav_path (str): Caminho para salvar o arquivo WAV gerado.

    Returns:
        bool: True se a conversão foi bem-sucedida, False caso contrário.
    """
    try:
        # Carrega o arquivo MIDI usando a biblioteca music21.
        sc = music21.converter.parse(midi_path)

        # Configurações para a síntese:
        sr = 16000  # Taxa de amostragem (frequência de amostragem em Hz).
        audio = np.array([])  # Inicializa o array de áudio vazio.

        # Itera sobre os elementos no arquivo MIDI.
        for element in sc.flat.notes:
            if isinstance(element, music21.note.Note):  # Se for uma nota musical:
                freq = element.pitch.frequency  # Frequência associada à nota.
                # Duração da nota baseada em seu valor e no andamento.
                dur = element.duration.quarterLength * (60 / sc.metronomeMarkBoundaries()[0][2].number)
                t = np.linspace(0, dur, int(sr * dur), False)  # Gera o tempo discreto.
                note_audio = 0.5 * np.sin(2 * np.pi * freq * t)  # Gera a onda senoidal.
                audio = np.concatenate((audio, note_audio))  # Adiciona a onda ao áudio total.
            elif isinstance(element, music21.note.Rest):  # Se for uma pausa:
                dur = element.duration.quarterLength * (60 / sc.metronomeMarkBoundaries()[0][2].number)
                t = np.linspace(0, dur, int(sr * dur), False)  # Gera a pausa no tempo.
                rest_audio = np.zeros_like(t)  # Cria uma sequência de zeros para representar silêncio.
                audio = np.concatenate((audio, rest_audio))  # Adiciona a pausa ao áudio total.

        # Normaliza o áudio para o intervalo [-1, 1], prevenindo distorção.
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio

        # Salva o áudio gerado no formato WAV.
        sf.write(wav_path, audio, sr)
        return True
    except Exception as e:
        # Captura erros durante a conversão.
        st.error(f"Erro na conversão de MIDI para WAV usando síntese simples: {e}")
        return False
def showScore(xml_string):
    """
    Renderiza a partitura musical usando OpenSheetMusicDisplay (OSMD) em um componente HTML do Streamlit.

    Args:
        xml_string (str): String contendo a partitura em formato MusicXML.

    Notas:
    - O OSMD é uma biblioteca de código aberto para renderizar partituras musicais diretamente em navegadores.
    - Aqui, utilizamos o Streamlit para integrar o OSMD como um componente HTML.
    """
    DIV_ID = "OSMD_div"  # ID do elemento HTML onde a partitura será renderizada.
    # HTML e script para carregar e renderizar a partitura.
    html_content = f"""
    <div id="{DIV_ID}">Carregando OpenSheetMusicDisplay...</div>
    <script>
    (function() {{
        // Verifica se o OSMD já foi carregado no navegador
        if (!window.opensheetmusicdisplay) {{
            var script = document.createElement('script');
            script.src = "https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@0.7.6/build/opensheetmusicdisplay.min.js";
            script.onload = function() {{
                initializeOSMD();
            }};
            document.head.appendChild(script);
        }} else {{
            initializeOSMD();
        }}

        function initializeOSMD() {{
            var osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("{DIV_ID}", {{
                drawingParameters: "compacttight"  // Configurações de desenho para layout compacto.
            }});
            osmd.load(`{xml_string}`).then(function() {{
                osmd.render();  // Renderiza a partitura no navegador.
            }});
        }}
    }})();
    </script>
    """
    components.html(html_content, height=600)  # Insere o conteúdo HTML no Streamlit.

def create_music_score(best_notes_and_rests, tempo, showScore, tmp_audio_path, soundfont_path):
    """
    Cria uma partitura musical a partir de uma lista de notas e pausas, renderiza no navegador e
    converte para áudio (MIDI e WAV).

    Args:
        best_notes_and_rests (list): Lista contendo as notas (e.g., "C4") e pausas ("Rest").
        tempo (int): Tempo da música em batidas por minuto (BPM).
        showScore (function): Função para renderizar a partitura (showScore).
        tmp_audio_path (str): Caminho temporário para salvar os arquivos MIDI e WAV.
        soundfont_path (str): Caminho para o arquivo SoundFont usado na conversão MIDI para WAV.

    Notas:
    - O `music21` é usado para criar e manipular a partitura.
    - As entradas inválidas são filtradas para evitar erros durante a criação.
    """
    if not best_notes_and_rests:
        st.error("Nenhuma nota ou pausa foi fornecida para criar a partitura.")
        return

    # Exibe as primeiras 20 notas ou pausas detectadas para depuração.
    st.write(f"**Notas e Pausas Detectadas (Primeiros 20):** {best_notes_and_rests[:20]}")

    # Filtra entradas inválidas para garantir que todas as notas e pausas sejam reconhecíveis.
    valid_notes = set(note_names + [f"{n}{o}" for n in note_names for o in range(0, 10)] + ["Rest"])
    invalid_entries = [note for note in best_notes_and_rests if note not in valid_notes]
    if invalid_entries:
        st.warning(f"Entradas inválidas detectadas e serão ignoradas: {invalid_entries}")
        best_notes_and_rests = [note for note in best_notes_and_rests if note in valid_notes]

    if not best_notes_and_rests:
        st.error("Todas as entradas fornecidas são inválidas. A partitura não pode ser criada.")
        return

    # Criação da partitura usando o music21.
    sc = music21.stream.Score()  # Objeto principal da partitura.
    part = music21.stream.Part()  # Parte da música (instrumento ou voz).
    sc.insert(0, part)  # Insere a parte na partitura.

    # Configurações padrão da partitura:
    # Adiciona a assinatura de tempo (4/4).
    time_signature = music21.meter.TimeSignature('4/4')
    part.insert(0, time_signature)

    # Define o andamento (tempo).
    tempo_mark = music21.tempo.MetronomeMark(number=tempo)
    part.insert(1, tempo_mark)

    # Adiciona a clave (clave de sol).
    clef_obj = music21.clef.TrebleClef()
    part.insert(2, clef_obj)

    # Define a tonalidade (Dó maior ou Lá menor).
    key_signature = music21.key.KeySignature(0)  # 0 indica ausência de sustenidos ou bemóis.
    part.insert(3, key_signature)

    # Inicia o primeiro compasso.
    measure = music21.stream.Measure(number=1)
    part.append(measure)

    # Configuração de durações:
    total_duration = 4.0  # Duração total de um compasso (4/4).
    current_duration = 0.0  # Duração atual preenchida no compasso.

    for snote in best_notes_and_rests:
        if snote == 'Rest':  # Adiciona pausas.
            note_obj = music21.note.Rest()
            note_obj.duration.type = 'quarter'  # Duração padrão: 1/4 de compasso.
        else:  # Adiciona notas musicais.
            try:
                note_obj = music21.note.Note(snote)
                note_obj.duration.type = 'quarter'  # Duração padrão: 1/4 de compasso.
            except music21.pitch.PitchException:
                st.warning(f"Nota inválida detectada e ignorada: {snote}")
                continue

        measure.append(note_obj)  # Adiciona a nota ou pausa ao compasso.
        current_duration += note_obj.duration.quarterLength

        # Cria um novo compasso quando o atual está completo.
        if current_duration >= total_duration:
            current_duration = 0.0
            measure = music21.stream.Measure(number=measure.number + 1)
            part.append(measure)

    # Completa o último compasso com pausas, se necessário.
    if current_duration > 0.0 and current_duration < total_duration:
        remaining = total_duration - current_duration
        rest_obj = music21.note.Rest()
        rest_obj.duration.quarterLength = remaining
        measure.append(rest_obj)

    # Verifica se a partitura está bem-formada.
    if sc.isWellFormedNotation():
        try:
            # Exporta a partitura para MusicXML em um arquivo temporário.
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xml') as tmp_musicxml:
                sc.write('musicxml', fp=tmp_musicxml.name)
                tmp_musicxml_path = tmp_musicxml.name

            # Lê o conteúdo do MusicXML e exibe a partitura.
            try:
                with open(tmp_musicxml_path, 'r', encoding='utf-8') as f:
                    xml_string = f.read()
            except Exception as e:
                st.error(f"Erro ao ler o arquivo MusicXML: {e}")
                return
            finally:
                os.remove(tmp_musicxml_path)  # Remove o arquivo temporário.

            # Renderiza a partitura usando o OpenSheetMusicDisplay.
            showScore(xml_string)

            # Gera um arquivo MIDI a partir da partitura.
            converted_audio_file_as_midi = tmp_audio_path[:-4] + '.mid'
            sc.write('midi', fp=converted_audio_file_as_midi)

            # Converte o MIDI gerado para WAV usando o FluidSynth.
            success = midi_to_wav(converted_audio_file_as_midi, tmp_audio_path[:-4] + '.wav', soundfont_path)

            if success:
                # Disponibiliza o arquivo WAV para download e reprodução.
                with open(tmp_audio_path[:-4] + '.wav', 'rb') as f:
                    wav_data = f.read()
                st.download_button(
                    label="Download do Arquivo WAV",
                    data=wav_data,
                    file_name=os.path.basename(tmp_audio_path[:-4] + '.wav'),
                    mime="audio/wav"
                )
                st.audio(wav_data, format='audio/wav')
                st.success("Arquivo WAV gerado, reproduzido e disponível para download.")
            else:
                st.error("Falha na conversão de MIDI para WAV.")
        except Exception as e:
            st.error(f"Erro ao processar a partitura: {e}")
    else:
        st.error("A partitura criada não está bem-formada. Verifique os dados de entrada.")
        try:
            sc_text = sc.show('text')  # Exibe a partitura em formato textual para depuração.
            st.write(sc_text)
        except Exception as e:
            st.error(f"Erro ao exibir a partitura para depuração: {e}")
def test_soundfont_conversion(soundfont_path):
    """
    Testa a conversão de um arquivo MIDI simples para WAV utilizando o SoundFont fornecido.

    Args:
        soundfont_path (str): Caminho para o arquivo SoundFont (.sf2) que será usado para a conversão.

    Funcionamento:
    - Cria um arquivo MIDI temporário contendo uma única nota musical.
    - Usa o SoundFont para converter o arquivo MIDI em áudio no formato WAV.
    - Reproduz o áudio resultante no Streamlit e indica se a conversão foi bem-sucedida.
    """
    try:
        # Criar dois arquivos temporários: um para MIDI e outro para WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as test_midi:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as test_wav:
                test_midi_path = test_midi.name  # Caminho do arquivo MIDI
                test_wav_path = test_wav.name  # Caminho do arquivo WAV

                # Criar uma partitura MIDI simples com uma única nota (Dó central)
                sc = music21.stream.Stream()  # Inicializa um fluxo de música
                n = music21.note.Note("C4")  # Nota "C4" (Dó central)
                n.duration.type = 'whole'  # Define a duração como "whole" (semibreve)
                sc.append(n)  # Adiciona a nota à partitura
                sc.write('midi', fp=test_midi_path)  # Salva a partitura como arquivo MIDI

                # Converter o MIDI para WAV usando o SoundFont fornecido
                success = midi_to_wav(test_midi_path, test_wav_path, soundfont_path)

                # Verificar se a conversão foi bem-sucedida e o arquivo WAV foi gerado
                if success and os.path.exists(test_wav_path):
                    st.success("Testes de conversão SoundFont: Sucesso!")  # Mensagem de sucesso no Streamlit
                    # Abrir o arquivo WAV e carregar os dados
                    with open(test_wav_path, 'rb') as f:
                        wav_data = f.read()
                    # Reproduzir o áudio gerado no Streamlit
                    st.audio(wav_data, format='audio/wav')
                else:
                    st.error("Testes de conversão SoundFont: Falhou.")  # Mensagem de erro se a conversão falhar

                # Remover os arquivos temporários após o teste
                os.remove(test_midi_path)
                os.remove(test_wav_path)
    except Exception as e:
        # Exibir mensagem de erro caso ocorra uma exceção
        st.error(f"Erro durante o teste de conversão SoundFont: {e}")


def get_class_name(class_index, class_map):
    """
    Retorna o nome da classe dado o índice.

    Args:
        class_index (int): Índice da classe predita.
        class_map (pd.DataFrame): DataFrame contendo o mapa de classes.

    Returns:
        str: Nome da classe correspondente.
    """
    try:
        class_name = class_map.iloc[class_index]['display_name']
        return class_name
    except IndexError:
        return "Índice de classe inválido."

def classify_new_audio(uploaded_audio):
    """
    Função para classificar um novo arquivo de áudio.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmp_audio:
            tmp_audio.write(uploaded_audio.read())
            tmp_audio_path = tmp_audio.name
    except Exception as e:
        st.error(f"Erro ao salvar o arquivo de áudio: {e}")
        return

    # Audição do Áudio Carregado
    try:
        sr_orig, wav_data = wavfile.read(tmp_audio_path)
        # Verificar se o áudio é mono e 16kHz
        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)
        if sr_orig != 16000:
            sr, wav_data = ensure_sample_rate(sr_orig, wav_data)
        else:
            sr = sr_orig
        # Normalizar
        if wav_data.dtype.kind == 'i':
            wav_data = wav_data / np.iinfo(wav_data.dtype).max
        elif wav_data.dtype.kind == 'f':
            if np.max(wav_data) > 1.0 or np.min(wav_data) < -1.0:
                wav_data = wav_data / np.max(np.abs(wav_data))
        # Converter para float32
        wav_data = wav_data.astype(np.float32)
        duration = len(wav_data) / sr
        st.write(f"**Taxa de Amostragem:** {sr} Hz")
        st.write(f"**Duração Total:** {duration:.2f}s")
        st.write(f"**Tamanho da Entrada:** {len(wav_data)} amostras")
        st.audio(wav_data, format='audio/wav', sample_rate=sr)
    except Exception as e:
        st.error(f"Erro ao processar o áudio para audição: {e}")
        wav_data = None

    # Extrair embeddings
    yamnet_model = load_yamnet_model()
    class_map = st.session_state.get('class_map', None)
    if class_map is None:
        # Carregar o mapa de classes se ainda não estiver carregado
        class_map_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
        class_map = load_class_map(class_map_url)
        st.session_state['class_map'] = class_map
    pred_class, embedding = extract_yamnet_embeddings(yamnet_model, tmp_audio_path)
    mfcc_features = extract_mfcc_features(tmp_audio_path)
    vibration_features = extract_vibration_features(tmp_audio_path)

    if embedding is not None and mfcc_features is not None and vibration_features is not None:
        # Obter o modelo treinado
        classifier = st.session_state['classifier']
        classes = st.session_state['classes']
        num_classes = len(classes)
        soundfont_path = st.session_state['soundfont_path']

        # Combinar todos os recursos
        combined_features = np.concatenate((embedding, mfcc_features, vibration_features))
        combined_features = combined_features.reshape(1, -1)  # Reshape para [1, n_features]

        # Normalizar os recursos (usando o mesmo scaler do treinamento)
        scaler = st.session_state.get('scaler', None)
        if scaler:
            combined_features = scaler.transform(combined_features)
        else:
            scaler = StandardScaler()
            combined_features = scaler.fit_transform(combined_features)
            st.session_state['scaler'] = scaler

        # Converter o embedding para tensor
        embedding_tensor = torch.tensor(combined_features, dtype=torch.float32).to(device)

        # Classificar
        classifier.eval()
        with torch.no_grad():
            output = classifier(embedding_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            class_idx = predicted.item()
            class_name = classes[class_idx]
            confidence_score = confidence.item()

        # Mapear o índice predito pelo YAMNet para o nome da classe
        class_name_yamnet = get_class_name(pred_class, class_map)

        # Exibir resultados
        st.write(f"**Classe Predita pelo YAMNet:** {pred_class} - {class_name_yamnet}")
        st.write(f"**Classe Predita pelo Classificador Personalizado:** {class_name}")
        st.write(f"**Confiança do Classificador Personalizado:** {confidence_score:.4f}")

        # Visualização do Áudio
        st.subheader("Visualização do Áudio")
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))

        # Plot da Forma de Onda
        ax[0].plot(wav_data)
        ax[0].set_title("Forma de Onda")
        ax[0].set_xlabel("Amostras")
        ax[0].set_ylabel("Amplitude")

        # Plot do Espectrograma
        S = librosa.feature.melspectrogram(y=wav_data, sr=sr, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax[1])
        fig.colorbar(ax[1].collections[0], ax=ax[1], format='%+2.0f dB')
        ax[1].set_title("Espectrograma Mel")

        st.pyplot(fig)
        plt.close(fig)

        # Detecção de Pitch com SPICE
        st.subheader("Detecção de Pitch com SPICE")
        # Carregar o modelo SPICE
        spice_model = hub.load("https://tfhub.dev/google/spice/2")

        # Normalizar as amostras de áudio
        audio_samples = wav_data / np.max(np.abs(wav_data)) if np.max(np.abs(wav_data)) != 0 else wav_data

        # Executar o modelo SPICE
        model_output = spice_model.signatures["serving_default"](tf.constant(audio_samples, tf.float32))

        pitch_outputs = model_output["pitch"].numpy().flatten()
        uncertainty_outputs = model_output["uncertainty"].numpy().flatten()

        # 'Uncertainty' basicamente significa a inversão da confiança.
        confidence_outputs = 1.0 - uncertainty_outputs

        # Plotar pitch e confiança
        fig_pitch, ax_pitch = plt.subplots(figsize=(20, 10))
        ax_pitch.plot(pitch_outputs, label='Pitch')
        ax_pitch.plot(confidence_outputs, label='Confiança')
        ax_pitch.legend(loc="lower right")
        ax_pitch.set_title("Pitch e Confiança com SPICE")
        st.pyplot(fig_pitch)
        plt.close(fig_pitch)

        # Remover pitches com baixa confiança (ajuste o limiar)
        confident_indices = np.where(confidence_outputs >= 0.8)[0]
        confident_pitches = pitch_outputs[confidence_outputs >= 0.8]
        confident_pitch_values_hz = [output2hz(p) for p in confident_pitches]

        # Plotar apenas pitches com alta confiança
        fig_confident_pitch, ax_confident_pitch = plt.subplots(figsize=(20, 10))
        ax_confident_pitch.scatter(confident_indices, confident_pitch_values_hz, c="r", label='Pitch Confiante')
        ax_confident_pitch.set_ylim([0, 2000])  # Ajustar conforme necessário
        ax_confident_pitch.set_title("Pitches Confidentes Detectados com SPICE")
        ax_confident_pitch.set_xlabel("Amostras")
        ax_confident_pitch.set_ylabel("Pitch (Hz)")
        ax_confident_pitch.legend()
        st.pyplot(fig_confident_pitch)
        plt.close(fig_confident_pitch)

        # Conversão de Pitches para Notas Musicais
        st.subheader("Notas Musicais Detectadas")
        # note_names e C0 já estão definidos globalmente

        def hz2offset(freq):
            # Medir o erro de quantização para uma única nota.
            if freq == 0:  # Silêncio sempre tem erro zero.
                return None
            # Nota quantizada.
            h = round(12 * math.log2(freq / C0))
            return 12 * math.log2(freq / C0) - h

        # Calcular o offset ideal
        offsets = [hz2offset(p) for p in confident_pitch_values_hz if p != 0]
        if offsets:
            ideal_offset = statistics.mean(offsets)
        else:
            ideal_offset = 0.0
        # Garantir que ideal_offset é float
        ideal_offset = float(ideal_offset)
        st.write(f"Ideal Offset: {ideal_offset:.4f}")

        # Função para quantizar previsões
        def quantize_predictions(group, ideal_offset):
            # Group values são ou 0, ou um pitch em Hz.
            non_zero_values = [v for v in group if v != 0]
            zero_values_count = len(group) - len(non_zero_values)

            # Criar um rest se 80% for silencioso, caso contrário, criar uma nota.
            if zero_values_count > 0.8 * len(group):
                # Interpretar como um rest. Contar cada nota descartada como um erro, ponderado um pouco pior que uma nota mal cantada (que 'custaria' 0.5).
                return 0.51 * len(non_zero_values), "Rest"
            else:
                # Interpretar como nota, estimando como média das previsões não-rest.
                h = round(
                    statistics.mean([
                        12 * math.log2(freq / C0) - ideal_offset for freq in non_zero_values
                    ])
                )
                octave = h // 12
                n = h % 12
                # Garantir que a nota está dentro do intervalo MIDI válido (0-127)
                midi_number = h + 60  # Adicionando 60 para centralizar em torno de C4
                if midi_number < 0 or midi_number > 127:
                    st.warning(f"Número MIDI {midi_number} fora do intervalo válido (0-127). Será ignorado.")
                    return float('inf'), "Rest"
                if n < 0 or n >= len(note_names):
                    st.warning(f"Índice de nota inválido: {n}. Nota será ignorada.")
                    return float('inf'), "Rest"
                note = note_names[n] + str(octave)
                # Erro de quantização é a diferença total da nota quantizada.
                error = sum([
                    abs(12 * math.log2(freq / C0) - ideal_offset - h)
                    for freq in non_zero_values
                ])
                return error, note

        # Agrupar pitches em notas (simplificação: usar uma janela deslizante)
        predictions_per_eighth = 40  # Aumentado de 20 para 40
        prediction_start_offset = 0  # Ajustar conforme necessário

        def get_quantization_and_error(pitch_outputs_and_rests, predictions_per_eighth,
                                       prediction_start_offset, ideal_offset):
            # Aplicar o offset inicial - podemos simplesmente adicionar o offset como rests.
            pitch_outputs_and_rests = [0] * prediction_start_offset + list(pitch_outputs_and_rests)
            # Coletar as previsões para cada nota (ou rest).
            groups = [
                pitch_outputs_and_rests[i:i + predictions_per_eighth]
                for i in range(0, len(pitch_outputs_and_rests), predictions_per_eighth)
            ]

            quantization_error = 0

            notes_and_rests = []
            for group in groups:
                error, note_or_rest = quantize_predictions(group, ideal_offset)
                quantization_error += error
                notes_and_rests.append(note_or_rest)

            return quantization_error, notes_and_rests

        # Obter a melhor quantização
        best_error = float("inf")
        best_notes_and_rests = None
        best_predictions_per_eighth = None

        for ppn in range(15, 35, 1):
            for pso in range(ppn):
                error, notes_and_rests = get_quantization_and_error(
                    confident_pitch_values_hz, ppn,
                    pso, ideal_offset
                )
                if error < best_error:
                    best_error = error
                    best_notes_and_rests = notes_and_rests
                    best_predictions_per_eighth = ppn

        # Remover rests iniciais e finais
        while best_notes_and_rests and best_notes_and_rests[0] == 'Rest':
            best_notes_and_rests = best_notes_and_rests[1:]
        while best_notes_and_rests and best_notes_and_rests[-1] == 'Rest':
            best_notes_and_rests = best_notes_and_rests[:-1]

        # Garantir que todas as notas e rests são strings
        best_notes_and_rests = [str(note) for note in best_notes_and_rests]

        st.write(f"Notas e Pausas Detectadas: {best_notes_and_rests}")

        # Estimar Tempo (BPM) usando Librosa
        try:
            tempo, beat_frames = librosa.beat.beat_track(y=wav_data, sr=sr)
            # Garantir que tempo é float e tratar possíveis arrays
            if isinstance(tempo, np.ndarray):
                if tempo.size == 1:
                    tempo = float(tempo.item())
                else:
                    st.warning(f"'tempo' é um array com múltiplos elementos: {tempo}")
                    tempo = float(tempo[0])  # Seleciona o primeiro elemento como exemplo
            elif isinstance(tempo, (int, float)):
                tempo = float(tempo)
            else:
                st.error(f"Tipo inesperado para 'tempo': {type(tempo)}")
                tempo = 120.0  # Valor padrão ou lidar de acordo com a lógica do seu aplicativo

            st.write(f"**BPM Inferido com Librosa:** {tempo:.2f}")
        except Exception as e:
            st.error(f"Erro ao estimar o tempo (BPM) do áudio: {e}")
            tempo = 120.0  # Valor padrão

        # Criar a partitura musical usando music21
        create_music_score(best_notes_and_rests, tempo, showScore, tmp_audio_path, soundfont_path)

def output2hz(pitch_output):
    # Constantes retiradas de https://tfhub.dev/google/spice/2
    PT_OFFSET = 25.58
    PT_SLOPE = 63.07
    FMIN = 10.0
    BINS_PER_OCTAVE = 12.0
    cqt_bin = pitch_output * PT_SLOPE + PT_OFFSET
    return FMIN * 2.0 ** (1.0 * cqt_bin / BINS_PER_OCTAVE)

def main():
    # Configurações da página - Deve ser chamado antes de qualquer outro comando do Streamlit
    st.set_page_config(page_title="Classificação de Áudio Avançada para Qualidade da Água", layout="wide")
    st.title("Classificação de Áudio Avançada para Avaliação da Qualidade e Temperatura da Água")
    st.write("""
    Este aplicativo permite treinar um classificador de áudio supervisionado utilizando o modelo **YAMNet** para extrair embeddings, **MFCCs** para capturar características espectrais, e análise de **vibração** para detectar variações sonoras e vibracionais da água. Além disso, incorpora o modelo **SPICE** para detecção de pitch, melhorando a classificação da qualidade potável, poluída e a temperatura da água.
    """)

    # Sidebar para parâmetros de treinamento e pré-processamento
    st.sidebar.header("Configurações")

    # Parâmetros de Treinamento
    st.sidebar.subheader("Parâmetros de Treinamento")
    epochs = st.sidebar.number_input("Número de Épocas:", min_value=1, max_value=500, value=100, step=1)
    learning_rate = st.sidebar.select_slider("Taxa de Aprendizagem:", options=[0.1, 0.01, 0.001, 0.0001], value=0.001)
    batch_size = st.sidebar.selectbox("Tamanho de Lote:", options=[8, 16, 32, 64, 128], index=2)
    l2_lambda = st.sidebar.number_input("Regularização L2 (Weight Decay):", min_value=0.0, max_value=0.1, value=0.01, step=0.01)
    patience = st.sidebar.number_input("Paciência para Early Stopping:", min_value=1, max_value=20, value=5, step=1)

    # Definir a seed com base na entrada do usuário
    seed = st.sidebar.number_input("Seed (número para tornar os resultados iguais sempre):", min_value=0, max_value=10000, value=42, step=1)
    set_seed(seed)  # Usar a seed escolhida pelo usuário

    # Opções de Data Augmentation
    st.sidebar.subheader("Data Augmentation")
    augment = st.sidebar.checkbox("Aplicar Data Augmentation")
    if augment:
        augmentation_methods = st.sidebar.multiselect(
            "Métodos de Data Augmentation:",
            options=["Adicionar Ruído", "Esticar Tempo", "Mudar Pitch"],
            default=["Adicionar Ruído", "Esticar Tempo"]
        )
        # Parâmetros adicionais para Data Augmentation
        rate = st.sidebar.slider("Taxa para Esticar Tempo:", min_value=0.5, max_value=2.0, value=1.2, step=0.1)
        n_steps = st.sidebar.slider("Passos para Mudar Pitch:", min_value=-12, max_value=12, value=3, step=1)
    else:
        augmentation_methods = []
        rate = 1.2
        n_steps = 3

    # Opções de Balanceamento de Classes
    st.sidebar.subheader("Balanceamento de Classes")
    balance_method = st.sidebar.selectbox(
        "Método de Balanceamento:",
        options=["Nenhum", "Oversample", "Undersample"],
        index=0
    )

    # Seção de Upload do SoundFont
    st.header("Upload do SoundFont (SF2)")
    st.write("""
    Para converter arquivos MIDI em WAV, é necessário um SoundFont (arquivo `.sf2`). Faça o upload do seu SoundFont aqui.
    """)
    uploaded_soundfont = st.file_uploader("Faça upload do arquivo SoundFont (.sf2)", type=["sf2"])

    if uploaded_soundfont is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".sf2") as tmp_sf2:
                tmp_sf2.write(uploaded_soundfont.read())
                soundfont_path = tmp_sf2.name
            st.success("SoundFont carregado com sucesso.")
        except Exception as e:
            st.error(f"Erro ao carregar o SoundFont: {e}")
            soundfont_path = None
    else:
        soundfont_path = None
        st.warning("Por favor, faça upload de um SoundFont (.sf2) para continuar.")

    if soundfont_path:
        # Seção de Teste do SoundFont
        st.header("Teste do SoundFont")
        st.write("""
        Clique no botão abaixo para testar a conversão de um MIDI simples para WAV usando o SoundFont carregado.
        """)
        if st.button("Executar Teste de Conversão SoundFont"):
            test_soundfont_conversion(soundfont_path)

        # Seção de Download e Preparação de Arquivos de Áudio
        st.header("Baixando e Preparando Arquivos de Áudio")
        st.write("""
        Você pode baixar arquivos de áudio de exemplo ou carregar seus próprios arquivos para começar.
        """)

        # Links para Download de Arquivos de Áudio de Exemplo
        st.subheader("Download de Arquivos de Áudio de Exemplo")
        sample_audio_1 = 'speech_whistling2.wav'
        sample_audio_2 = 'miaow_16k.wav'
        sample_audio_1_url = "https://storage.googleapis.com/audioset/speech_whistling2.wav"
        sample_audio_2_url = "https://storage.googleapis.com/audioset/miaow_16k.wav"

        # Função para Baixar Arquivos
        def download_audio(url, filename):
            try:
                response = requests.get(url)
                response.raise_for_status()  # Verifica se a requisição foi bem-sucedida
                with open(filename, 'wb') as f:
                    f.write(response.content)
                st.success(f"Arquivo `{filename}` baixado com sucesso.")
            except Exception as e:
                st.error(f"Erro ao baixar {filename}: {e}")

        # Botões de Download
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"Baixar {sample_audio_1}"):
                download_audio(sample_audio_1_url, sample_audio_1)
        with col2:
            if st.button(f"Baixar {sample_audio_2}"):
                download_audio(sample_audio_2_url, sample_audio_2)

        # Audição de Arquivos de Áudio de Exemplo
        st.subheader("Audição de Arquivos de Áudio de Exemplo")
        uploaded_file_example = st.selectbox("Selecione um arquivo de áudio de exemplo para ouvir:", options=["Nenhum", sample_audio_1, sample_audio_2])

        if uploaded_file_example != "Nenhum" and os.path.exists(uploaded_file_example):
            try:
                sr_orig, wav_data = wavfile.read(uploaded_file_example)
                # Verificar se o áudio é mono e 16kHz
                if wav_data.ndim > 1:
                    wav_data = wav_data.mean(axis=1)
                if sr_orig != 16000:
                    sr, wav_data = ensure_sample_rate(sr_orig, wav_data)
                else:
                    sr = sr_orig
                # Normalizar
                if wav_data.dtype.kind == 'i':
                    wav_data = wav_data / np.iinfo(wav_data.dtype).max
                elif wav_data.dtype.kind == 'f':
                    if np.max(wav_data) > 1.0 or np.min(wav_data) < -1.0:
                        wav_data = wav_data / np.max(np.abs(wav_data))
                # Converter para float32
                wav_data = wav_data.astype(np.float32)
                duration = len(wav_data) / sr
                st.write(f"**Taxa de Amostragem:** {sr} Hz")
                st.write(f"**Duração Total:** {duration:.2f}s")
                st.write(f"**Tamanho da Entrada:** {len(wav_data)} amostras")
                st.audio(wav_data, format='audio/wav', sample_rate=sr)
            except Exception as e:
                st.error(f"Erro ao processar o arquivo de áudio: {e}")
        elif uploaded_file_example != "Nenhum":
            st.warning("Arquivo de áudio não encontrado. Por favor, baixe o arquivo antes de tentar ouvir.")

        # Upload de dados supervisionados
        st.header("Upload de Dados Supervisionados")
        st.write("""
        Envie um arquivo ZIP contendo subpastas com arquivos de áudio organizados por classe. Por exemplo:
        """)
        st.write("""
        ```
        dados/
            agua_quente/
                audio1.wav
                audio2.wav
            agua_gelada/
                audio3.wav
                audio4.wav
        ```
        """)
        uploaded_zip = st.file_uploader("Faça upload do arquivo ZIP com os dados de áudio supervisionados", type=["zip"])

        if uploaded_zip is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "uploaded.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.read())
                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(tmpdir)
                    st.success("Arquivo ZIP extraído com sucesso.")
                except zipfile.BadZipFile:
                    st.error("O arquivo enviado não é um ZIP válido.")
                    st.stop()

                # Verificar estrutura de diretórios
                classes = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
                if len(classes) < 2:
                    st.error("O arquivo ZIP deve conter pelo menos duas subpastas, cada uma representando uma classe.")
                    st.stop()
                else:
                    st.success(f"Classes encontradas: {classes}")
                    # Contar arquivos por classe
                    class_counts = {}
                    for cls in classes:
                        cls_dir = os.path.join(tmpdir, cls)
                        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
                        class_counts[cls] = len(files)
                    st.write("**Contagem de arquivos por classe:**")
                    st.write(class_counts)

                    # Preparar dados para treinamento
                    st.header("Preparando Dados para Treinamento")
                    yamnet_model = load_yamnet_model()
                    st.write("Modelo YAMNet carregado.")

                    # Carregar o mapa de classes do YAMNet
                    class_map_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
                    class_map = load_class_map(class_map_url)
                    st.session_state['class_map'] = class_map  # Armazenar no session_state

                    embeddings = []
                    labels = []
                    mfccs = []
                    vibrations = []
                    label_mapping = {cls: idx for idx, cls in enumerate(classes)}

                    total_files = sum(class_counts.values())
                    processed_files = 0
                    progress_bar = st.progress(0)

                    scaler = StandardScaler()
                    for cls in classes:
                        cls_dir = os.path.join(tmpdir, cls)
                        audio_files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
                        for audio_file in audio_files:
                            pred_class, embedding = extract_yamnet_embeddings(yamnet_model, audio_file)
                            mfcc_feature = extract_mfcc_features(audio_file)
                            vibration_feature = extract_vibration_features(audio_file)
                            if embedding is not None and mfcc_feature is not None and vibration_feature is not None:
                                if augment:
                                    # Carregar o áudio usando librosa para aplicar augmentations
                                    try:
                                        waveform, sr = librosa.load(audio_file, sr=16000, mono=True)
                                        augmented_waveforms = perform_data_augmentation(waveform, sr, augmentation_methods, rate=rate, n_steps=n_steps)
                                        for aug_waveform in augmented_waveforms:
                                            # Salvar temporariamente o áudio aumentado para processar
                                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                                                sf.write(temp_audio.name, aug_waveform, sr)
                                                aug_pred_class, aug_embedding = extract_yamnet_embeddings(yamnet_model, temp_audio.name)
                                                aug_mfcc = extract_mfcc_features(temp_audio.name)
                                                aug_vibration = extract_vibration_features(temp_audio.name)
                                                if aug_embedding is not None and aug_mfcc is not None and aug_vibration is not None:
                                                    embeddings.append(aug_embedding)
                                                    mfccs.append(aug_mfcc)
                                                    vibrations.append(aug_vibration)
                                                    labels.append(label_mapping[cls])
                                                os.remove(temp_audio.name)
                                    except Exception as e:
                                        st.warning(f"Erro ao aplicar data augmentation no arquivo {audio_file}: {e}")
                                else:
                                    embeddings.append(embedding)
                                    mfccs.append(mfcc_feature)
                                    vibrations.append(vibration_feature)
                                    labels.append(label_mapping[cls])
                            processed_files += 1
                            progress_bar.progress(processed_files / total_files)

                    # Verificações após a extração
                    if len(embeddings) == 0:
                        st.error("Nenhum embedding foi extraído. Verifique se os arquivos de áudio estão no formato correto e se o YAMNet está funcionando corretamente.")
                        st.stop()

                    # Verificar se todos os embeddings têm o mesmo tamanho
                    embedding_shapes = [emb.shape for emb in embeddings]
                    unique_shapes = set(embedding_shapes)
                    if len(unique_shapes) != 1:
                        st.error(f"Embeddings têm tamanhos inconsistentes: {unique_shapes}")
                        st.stop()

                    # Converter para array NumPy
                    try:
                        embeddings = np.array(embeddings)
                        mfccs = np.array(mfccs)
                        vibrations = np.array(vibrations)
                        labels = np.array(labels)
                        st.write(f"Embeddings convertidos para array NumPy: Forma = {embeddings.shape}")
                        st.write(f"MFCCs convertidos para array NumPy: Forma = {mfccs.shape}")
                        st.write(f"Características de Vibração convertidas para array NumPy: Forma = {vibrations.shape}")
                    except ValueError as ve:
                        st.error(f"Erro ao converter embeddings para array NumPy: {ve}")
                        st.stop()

                    # Concatenar todos os recursos
                    combined_features = np.concatenate((embeddings, mfccs, vibrations), axis=1)
                    st.write(f"Características combinadas: Forma = {combined_features.shape}")

                    # Normalizar os recursos
                    combined_features = scaler.fit_transform(combined_features)
                    st.write("Características normalizadas com StandardScaler.")

                    # Análise de Dados com DataFrames
                    st.header("Análise de Dados")
                    st.write("**Estatísticas Descritivas das Características Combinadas:**")
                    combined_df = pd.DataFrame(combined_features)
                    st.dataframe(combined_df.describe())

                    st.write("**Distribuição das Classes:**")
                    class_distribution = pd.Series(labels).value_counts().rename(index={v: k for k, v in label_mapping.items()})
                    st.bar_chart(class_distribution)

                    # Plotagem dos Embeddings
                    st.write("**Visualização dos Embeddings com PCA:**")
                    plot_embeddings(combined_features, labels, classes)

                    # Balanceamento de Classes
                    if balance_method != "Nenhum":
                        st.write(f"Aplicando balanceamento de classes: {balance_method}")
                        embeddings_bal, labels_bal = balance_classes(combined_features, labels, balance_method)
                        # Contar novamente após balanceamento
                        balanced_counts = {cls: 0 for cls in classes}
                        for label in labels_bal:
                            cls = [k for k, v in label_mapping.items() if v == label][0]
                            balanced_counts[cls] += 1
                        st.write(f"**Contagem de classes após balanceamento:**")
                        st.write(balanced_counts)
                    else:
                        embeddings_bal, labels_bal = combined_features, labels

                    # Ajustar n_splits para não exceder o mínimo de amostras por classe
                    min_class_size = min([count for count in class_counts.values()])
                    k_folds = min(10, min_class_size)  # Ajustar n_splits dinamicamente
                    if k_folds < 2:
                        st.error(f"Número de folds insuficiente para realizar validação cruzada (k_folds={k_folds}).")
                        st.stop()

                    st.header("Treinamento com Validação Cruzada")
                    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                    fold_results = []

                    for fold, (train_index, val_index) in enumerate(skf.split(embeddings_bal, labels_bal)):
                        st.write(f"### Fold {fold+1}/{k_folds}")
                        X_train_fold, X_val_fold = embeddings_bal[train_index], embeddings_bal[val_index]
                        y_train_fold, y_val_fold = labels_bal[train_index], labels_bal[val_index]

                        st.write(f"Treino: {len(X_train_fold)} amostras | Validação: {len(X_val_fold)} amostras")

                        # Treinar o classificador
                        classifier = train_audio_classifier(
                            X_train_fold, 
                            y_train_fold, 
                            X_val_fold, 
                            y_val_fold, 
                            input_dim=combined_features.shape[1], 
                            num_classes=len(classes), 
                            classes=classes,  # Passando 'classes' como argumento
                            epochs=epochs, 
                            learning_rate=learning_rate, 
                            batch_size=batch_size, 
                            l2_lambda=l2_lambda, 
                            patience=patience
                        )

                        # Avaliar e salvar os resultados
                        classifier.eval()
                        with torch.no_grad():
                            X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
                            y_val_tensor = torch.tensor(y_val_fold, dtype=torch.long).to(device)
                            outputs = classifier(X_val_tensor)
                            _, preds = torch.max(outputs, 1)
                            preds = preds.cpu().numpy()
                            y_true = y_val_fold

                            # Métricas
                            report = classification_report(y_true, preds, target_names=classes, zero_division=0, output_dict=True)
                            cm = confusion_matrix(y_true, preds)

                            # Salvar resultados do fold
                            fold_results.append({
                                'fold': fold+1,
                                'report': report,
                                'confusion_matrix': cm
                            })

                    # Agregar Resultados dos Folds
                    st.header("Resultados da Validação Cruzada")
                    for result in fold_results:
                        st.write(f"#### Fold {result['fold']}")
                        st.write("**Relatório de Classificação:**")
                        st.write(pd.DataFrame(result['report']).transpose())

                        st.write("**Matriz de Confusão:**")
                        cm_df = pd.DataFrame(result['confusion_matrix'], index=classes, columns=classes)
                        st.dataframe(cm_df)

                    # Plotagem da Média das Métricas
                    st.header("Média das Métricas de Avaliação")
                    avg_report = {}
                    for key in fold_results[0]['report'].keys():
                        if key not in ['accuracy', 'macro avg', 'weighted avg']:
                            avg_report[key] = {
                                'precision': np.mean([r['report'][key]['precision'] for r in fold_results]),
                                'recall': np.mean([r['report'][key]['recall'] for r in fold_results]),
                                'f1-score': np.mean([r['report'][key]['f1-score'] for r in fold_results]),
                                'support': np.sum([r['report'][key]['support'] for r in fold_results])
                            }
                    avg_report_df = pd.DataFrame(avg_report).transpose()
                    st.write("**Média de Precision, Recall e F1-Score por Classe:**")
                    st.dataframe(avg_report_df)

                    # Salvar o classificador no estado do Streamlit
                    st.session_state['classifier'] = classifier
                    st.session_state['classes'] = classes
                    st.session_state['soundfont_path'] = soundfont_path
                    st.session_state['scaler'] = scaler

                    # Exibir mensagem de conclusão do treinamento
                    st.success("Treinamento do classificador concluído.")

                    # Opção para download do modelo treinado
                    buffer = io.BytesIO()
                    torch.save(classifier.state_dict(), buffer)
                    buffer.seek(0)
                    st.download_button(
                        label="Download do Modelo Treinado",
                        data=buffer,
                        file_name="audio_classifier.pth",
                        mime="application/octet-stream"
                    )

                    # Opção para download do mapeamento de classes
                    class_mapping = "\n".join([f"{cls}:{idx}" for cls, idx in label_mapping.items()])
                    st.download_button(
                        label="Download do Mapeamento de Classes",
                        data=class_mapping,
                        file_name="classes_mapping.txt",
                        mime="text/plain"
                    )

    # Classificação de Novo Áudio
    if 'classifier' in st.session_state and 'classes' in st.session_state and 'soundfont_path' in st.session_state:
        st.header("Classificação de Novo Áudio")
        st.write("""
        **Envie um arquivo de áudio para ser classificado pelo modelo treinado.**
        
        **Explicação para Leigos:**
        - **O que você faz:** Envie um arquivo de áudio (como um canto, fala ou som ambiente) para que o modelo identifique a qual categoria ele pertence.
        - **O que acontece:** O aplicativo analisará o áudio, determinará a classe mais provável e mostrará a confiança dessa previsão. Além disso, você poderá visualizar a forma de onda, o espectrograma e as notas musicais detectadas.
        
        **Explicação para Técnicos:**
        - **Processo:** O áudio carregado é pré-processado para garantir a taxa de amostragem de 16kHz e convertido para mono. Em seguida, os embeddings são extraídos usando o modelo YAMNet, MFCCs são calculados para capturar características espectrais, e características vibracionais são extraídas via FFT. O classificador treinado em PyTorch utiliza esses recursos combinados para prever a classe do áudio, fornecendo uma pontuação de confiança baseada na função softmax.
        - **Detecção de Pitch:** Utilizando o modelo SPICE, o aplicativo realiza a detecção de pitch no áudio, convertendo os valores normalizados para Hz e quantizando-os em notas musicais utilizando a biblioteca `music21`. As notas detectadas são visualizadas e podem ser convertidas em um arquivo MIDI para reprodução.
        """)
        uploaded_audio = st.file_uploader("Faça upload do arquivo de áudio para classificação", type=["wav", "mp3", "ogg", "flac"])

        if uploaded_audio is not None:
            classify_new_audio(uploaded_audio)

    # Documentação e Agradecimentos
    st.write("### Documentação dos Procedimentos")
    st.write("""
    1. **Upload do SoundFont (SF2):** Faça o upload do seu arquivo SoundFont (`.sf2`) para permitir a conversão de MIDI para WAV.
    
    2. **Teste do SoundFont:** Execute o teste de conversão para garantir que o SoundFont está funcionando corretamente.
    
    3. **Baixando e Preparando Arquivos de Áudio:** Você pode baixar arquivos de áudio de exemplo ou carregar seus próprios arquivos para começar.
    
    4. **Upload de Dados Supervisionados:** Envie um arquivo ZIP contendo subpastas, onde cada subpasta representa uma classe com seus respectivos arquivos de áudio.
    
    5. **Data Augmentation:** Se selecionado, aplica métodos de data augmentation como adição de ruído, estiramento de tempo e mudança de pitch nos dados de treinamento. Você pode ajustar os parâmetros `rate` e `n_steps` para controlar a intensidade dessas transformações.
    
    6. **Balanceamento de Classes:** Se selecionado, aplica métodos de balanceamento como oversampling (SMOTE) ou undersampling para tratar classes desbalanceadas.
    
    7. **Extração de Embeddings:** Utilizamos o YAMNet para extrair embeddings dos arquivos de áudio enviados, além de MFCCs e características vibracionais para uma análise mais detalhada.
    
    8. **Treinamento com Validação Cruzada:** Com os embeddings extraídos e após as opções de data augmentation e balanceamento, treinamos um classificador utilizando validação cruzada para uma avaliação mais robusta.
    
    9. **Análise de Dados:** Visualize estatísticas descritivas, distribuição das classes e plotagens dos embeddings para melhor compreensão dos dados.
    
    10. **Resultados da Validação Cruzada:** Avalie o desempenho do modelo através de relatórios de classificação, matrizes de confusão e curvas ROC para cada fold.
    
    11. **Download dos Resultados:** Após o treinamento, você poderá baixar o modelo treinado e o mapeamento de classes.
    
    12. **Classificação de Novo Áudio:** Após o treinamento, você pode enviar um novo arquivo de áudio para ser classificado pelo modelo treinado. O aplicativo exibirá a classe predita pelo YAMNet, a classe predita pelo classificador personalizado, a confiança da predição, visualizará a forma de onda e o espectrograma do áudio carregado, realizará a detecção de pitch com SPICE e converterá as notas detectadas em uma partitura musical que poderá ser baixada e reproduzida.
    
    **Exemplo de Estrutura de Diretórios para Upload:**
    ```
    dados/
        agua_quente/
            audio1.wav
            audio2.wav
        agua_gelada/
            audio3.wav
            audio4.wav
    ```
    """)

    st.write("### Agradecimentos")
    st.write("""
    Desenvolvido por Marcelo Claro.
    
    - **Contato**: marceloclaro@gmail.com
    - **Whatsapp**: (88)98158-7145
    - **Instagram**: [marceloclaro.geomaker](https://www.instagram.com/marceloclaro.geomaker/)
    """)

if __name__ == "__main__":
    main()

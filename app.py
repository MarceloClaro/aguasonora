# app.py
import streamlit as st
import os
import random
import zipfile
import librosa
import librosa.display as ld
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import soundfile as sf
import gc

# Configurações para visualizações
sns.set(style='whitegrid', context='notebook')

# ==================== CONTROLE DE REPRODUTIBILIDADE ====================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==================== FUNÇÕES DE PROCESSAMENTO ====================

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

augment_transform = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    get_shift_transform(),
])

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
        augmented_data = augment_transform(samples=data, sample_rate=sr)
        return augmented_data
    except Exception as e:
        st.error(f"Erro ao aplicar Data Augmentation: {e}")
        return data  # Retorna o original em caso de erro

def plot_waveform(data, sr, title="Waveform"):
    """
    Plota a forma de onda do áudio.

    Parameters:
    - data (np.ndarray): Sinal de áudio.
    - sr (int): Taxa de amostragem.
    - title (str): Título do gráfico.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(title)
    ld.waveshow(data, sr=sr, ax=ax)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

def plot_frequency_spectrum(data, sr, title="Espectro de Frequências"):
    """
    Plota o espectro de frequências do áudio.

    Parameters:
    - data (np.ndarray): Sinal de áudio.
    - sr (int): Taxa de amostragem.
    - title (str): Título do gráfico.
    """
    N = len(data)
    fft = np.fft.fft(data)
    fft = np.abs(fft[:N//2])  # Apenas a metade positiva do espectro
    freqs = np.fft.fftfreq(N, 1/sr)[:N//2]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(title)
    ax.plot(freqs, fft)
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    st.pyplot(fig)

def plot_spectrogram(data, sr, title="Spectrograma (STFT)"):
    """
    Plota o espectrograma (STFT) do áudio.

    Parameters:
    - data (np.ndarray): Sinal de áudio.
    - sr (int): Taxa de amostragem.
    - title (str): Título do gráfico.
    """
    D = np.abs(librosa.stft(data))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(title)
    mappable = ld.specshow(DB, sr=sr, x_axis='time', y_axis='hz', cmap='magma', ax=ax)
    plt.colorbar(mappable=mappable, ax=ax, format='%+2.0f dB')
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Frequência (Hz)")
    st.pyplot(fig)

def plot_mfcc(data, sr, title="Spectrograma (MFCC)"):
    """
    Plota o espectrograma de MFCC do áudio.

    Parameters:
    - data (np.ndarray): Sinal de áudio.
    - sr (int): Taxa de amostragem.
    - title (str): Título do gráfico.
    """
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    mfccs_db = librosa.amplitude_to_db(np.abs(mfccs))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(title)
    mappable = ld.specshow(mfccs_db, x_axis='time', y_axis='mel', cmap='Spectral', sr=sr, ax=ax)
    plt.colorbar(mappable=mappable, ax=ax, format='%+2.f dB')
    st.pyplot(fig)

def plot_class_probabilities(class_probs, title="Probabilidades das Classes"):
    """
    Plota as probabilidades das classes em um gráfico de barras.

    Parameters:
    - class_probs (dict): Dicionário com as probabilidades de cada classe.
    - title (str): Título do gráfico.
    """
    classes = list(class_probs.keys())
    probs = list(class_probs.values())

    plt.figure(figsize=(12, 6))
    sns.barplot(x=classes, y=probs, palette='viridis')
    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Probabilidade")
    plt.ylim(0, 1)  # Probabilidades entre 0 e 1
    plt.show()
    st.pyplot(plt)

# ==================== FUNÇÃO PARA PREPARAR OS DADOS ====================

def prepare_data(uploaded_files):
    """
    Prepara os dados a partir dos arquivos enviados.

    Parameters:
    - uploaded_files (dict): Dicionário com os arquivos carregados.

    Returns:
    - df (pd.DataFrame): DataFrame contendo caminhos dos arquivos e suas classes.
    - base_path (str): Caminho base onde os arquivos foram extraídos.
    """
    file_paths = []
    labels = []
    temp_dir = "./temp_dataset"

    # Limpa o diretório temporário antes de extrair novos arquivos
    if os.path.exists(temp_dir):
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    else:
        os.makedirs(temp_dir, exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        if file.name.lower().endswith('.zip'):
            # Extrai o arquivo ZIP
            try:
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                st.success(f"Arquivo ZIP {file.name} extraído com sucesso.")
            except zipfile.BadZipFile:
                st.error(f"Erro: O arquivo {file.name} não é um ZIP válido.")
        else:
            # Salva arquivos individuais
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            st.success(f"Arquivo {file.name} carregado com sucesso.")

    # Define base_path como temp_dir
    base_path = temp_dir

    # Verifica se há subpastas (classes) dentro de base_path
    categories = [c for c in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, c))]

    if len(categories) == 0:
        # Listar a estrutura para depuração
        st.error("Nenhuma subpasta de classes encontrada dentro do ZIP. Verifique a estrutura do seu arquivo ZIP.")
        for root, dirs, files in os.walk(temp_dir):
            level = root.replace(temp_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            st.write(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                st.write(f"{sub_indent}{f}")
        return None, None

    st.write("Classes encontradas:", categories)

    # Coleta os caminhos dos arquivos e suas classes
    for cat in categories:
        cat_path = os.path.join(base_path, cat)
        # Inclui múltiplos formatos de áudio, se necessário
        files_in_cat = [f for f in os.listdir(cat_path) if f.lower().endswith(('.wav', '.mp3', '.flac'))]
        st.write(f"Classe '{cat}': {len(files_in_cat)} arquivos encontrados.")
        for file_name in files_in_cat:
            full_path = os.path.join(cat_path, file_name)
            file_paths.append(full_path)
            labels.append(cat)

    df = pd.DataFrame({'file_path': file_paths, 'class': labels})
    st.write(f"Total de amostras: {len(df)}")
    st.write("Primeiras amostras:", df.head())

    return df, base_path

# ==================== FUNÇÃO PARA TREINAR O MODELO ====================

def train_model(X, y, class_weight_dict):
    """
    Treina o modelo CNN.

    Parameters:
    - X (np.ndarray): Features de entrada.
    - y (np.ndarray): Labels de entrada.
    - class_weight_dict (dict): Pesos das classes para balanceamento.

    Returns:
    - model (tf.keras.Model): Modelo treinado.
    - history (tf.keras.callbacks.History): Histórico do treinamento.
    """
    # Definir a arquitetura da CNN
    model = Sequential([
        Conv1D(64, kernel_size=10, activation='relu', input_shape=(X.shape[1], 1)),
        Dropout(0.4),
        MaxPooling1D(pool_size=4),
        Conv1D(128, kernel_size=10, activation='relu', padding='same'),
        Dropout(0.4),
        MaxPooling1D(pool_size=4),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(y.shape[1], activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Resumo do modelo
    st.write("Resumo do modelo:")
    st.text(model.summary())

    # Definição dos parâmetros de treinamento
    num_epochs = st.sidebar.number_input("Número de Épocas", min_value=10, max_value=500, value=200)
    batch_size = st.sidebar.number_input("Tamanho do Batch", min_value=16, max_value=256, value=32)

    # Diretório para salvar o modelo
    save_dir = 'saved_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Definição dos callbacks
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(save_dir, 'model_agua_augmented.keras'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )

    earlystop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Treinamento do modelo
    with st.spinner('Treinando o modelo...'):
        history = model.fit(
            X, y,
            epochs=num_epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[checkpointer, earlystop],
            class_weight=class_weight_dict,
            verbose=1
        )
    st.success('Treinamento concluído!')

    return model, history

# ==================== FUNÇÃO PARA CLASSIFICAR NOVO ÁUDIO ====================

def classify_audio(model, labelencoder, audio_file_path):
    """
    Classifica um novo arquivo de áudio.

    Parameters:
    - model (tf.keras.Model): Modelo treinado.
    - labelencoder (LabelEncoder): LabelEncoder ajustado.
    - audio_file_path (str): Caminho para o arquivo de áudio.

    Returns:
    - pred_label (str): Classe prevista.
    - confidence (float): Grau de confiança.
    - class_probs (dict): Probabilidades das classes.
    """
    data, sr = load_audio(audio_file_path, sr=None)
    if data is None:
        st.error("Erro ao carregar o áudio para classificação.")
        return None, None, None

    features = extract_features(data, sr)
    if features is None:
        st.error("Erro ao extrair features do áudio para classificação.")
        return None, None, None

    features = features.reshape(1, -1, 1)

    prediction = model.predict(features)
    pred_class = np.argmax(prediction, axis=1)
    pred_label = labelencoder.inverse_transform(pred_class)
    confidence = prediction[0][pred_class][0]
    class_probs = {labelencoder.classes_[i]: float(prediction[0][i]) for i in range(len(labelencoder.classes_))}

    return pred_label[0], confidence, class_probs

# ==================== INÍCIO DA APLICAÇÃO STREAMLIT ====================

def main():
    st.title('Classificação de Sons de Água Vibrando em Copo de Vidro com CNN')

    st.sidebar.header('Configurações')
    st.sidebar.markdown('### Upload de Dados para Treinamento')

    uploaded_files = st.sidebar.file_uploader("Carregue arquivos ZIP contendo as pastas das classes ou arquivos de áudio individuais", type=['zip', 'wav', 'mp3', 'flac'], accept_multiple_files=True)
    train_button = st.sidebar.button('Treinar Modelo')

    st.header('Carregamento e Pré-processamento de Dados')
    if uploaded_files:
        df, base_path = prepare_data(uploaded_files)
        if df is not None:
            # Extrair features e preparar dados
            st.write("Extraindo features...")
            X = []
            y_valid = []
            for i, row in df.iterrows():
                file = row['file_path']
                data, sr = load_audio(file, sr=None)
                if data is not None:
                    features = extract_features(data, sr)
                    if features is not None:
                        X.append(features)
                        y_valid.append(row['class'])
            X = np.array(X)
            if len(X) == 0:
                st.error("Nenhuma feature foi extraída. Verifique seus arquivos de áudio.")
                return

            # Codificar as labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_valid)
            y_encoded = to_categorical(y_encoded)

            # Divisão dos dados
            st.write("Dividindo os dados em treinamento e teste...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
            )

            # Data Augmentation
            st.write("Aplicando Data Augmentation no conjunto de treinamento...")
            augment_factor = 10
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

            # Combinar dados originais e aumentados
            X_train_combined = np.concatenate((X_train, X_train_augmented), axis=0)
            y_train_combined = np.concatenate((y_train, y_train_augmented), axis=0)

            # Divisão em treino final e validação
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train_combined, y_train_combined, test_size=0.1, random_state=SEED, stratify=y_train_combined
            )

            # Reshape para Conv1D
            X_train_final = X_train_final.reshape((X_train_final.shape[0], X_train_final.shape[1], 1))
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            # Cálculo de class weights
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train_final.argmax(axis=1)),
                y=y_train_final.argmax(axis=1)
            )
            class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

            st.write("Dados preparados para o treinamento.")

            if train_button:
                model, history = train_model(X_train_final, y_train_final, class_weight_dict)

                # Salvar o modelo
                st.write("Modelo salvo em 'saved_models/model_agua_augmented.keras'.")

                # Avaliação do Modelo
                st.header('Avaliação do Modelo')
                score_train = model.evaluate(X_train_final, y_train_final, verbose=0)
                score_val = model.evaluate(X_val, y_val, verbose=0)
                score_test = model.evaluate(X_test, y_test, verbose=0)

                st.write(f"Acurácia Treino: {score_train[1]*100:.2f}%")
                st.write(f"Acurácia Validação: {score_val[1]*100:.2f}%")
                st.write(f"Acurácia Teste: {score_test[1]*100:.2f}%")

                # Predições no conjunto de teste
                y_pred = model.predict(X_test)
                y_pred_classes = y_pred.argmax(axis=1)
                y_true = y_test.argmax(axis=1)

                # Matriz de Confusão
                cm = confusion_matrix(y_true, y_pred_classes)
                cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
                fig, ax = plt.subplots(figsize=(12,8))
                sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title("Matriz de Confusão")
                ax.set_xlabel("Classe Prevista")
                ax.set_ylabel("Classe Real")
                st.pyplot(fig)

                # Relatório de Classificação
                report = classification_report(y_true, y_pred_classes, target_names=le.classes_, zero_division=0)
                st.text("Relatório de Classificação:")
                st.text(report)

                # Limpeza de memória
                del X_train, X_train_augmented, y_train, y_train_augmented
                gc.collect()

    else:
        st.write("Nenhum arquivo carregado. Por favor, carregue arquivos ZIP ou de áudio para começar.")

    # ==================== SEÇÃO DE CLASSIFICAÇÃO DE NOVO ÁUDIO ====================
    st.header('Classificação de Novo Áudio')
    uploaded_audio = st.file_uploader("Carregue um arquivo de áudio para classificar", type=['wav', 'mp3', 'flac'])

    if uploaded_audio and train_button:
        audio_file = uploaded_audio.name
        with open(audio_file, "wb") as f:
            f.write(uploaded_audio.getbuffer())
        st.success(f"Arquivo {audio_file} carregado com sucesso.")

        # Carregar o modelo salvo
        model_path = os.path.join('saved_models', 'model_agua_augmented.keras')
        if os.path.exists(model_path):
            loaded_model = tf.keras.models.load_model(model_path)
            st.success("Modelo carregado com sucesso.")
        else:
            st.error("Modelo não encontrado. Por favor, treine o modelo primeiro.")
            loaded_model = None

        if loaded_model:
            # Classificar o áudio
            pred_label, confidence, class_probs = classify_audio(loaded_model, le, audio_file)

            if pred_label:
                st.write(f"**Classe Prevista:** {pred_label}")
                st.write(f"**Grau de Confiança:** {confidence * 100:.2f}%")

                st.write("**Probabilidades das Classes:**")
                for cls, prob in class_probs.items():
                    st.write(f"{cls}: {prob * 100:.2f}%")

                # Plotar Probabilidades das Classes
                plot_class_probabilities(class_probs, title="Probabilidades das Classes")

                # Visualizações do Áudio Classificado
                data, sr = load_audio(audio_file, sr=None)
                if data is not None:
                    st.write("### Visualizações do Áudio Classificado")
                    plot_waveform(data, sr, title=f"Waveform - {pred_label}")
                    plot_frequency_spectrum(data, sr, title=f"Espectro de Frequências - {pred_label}")
                    plot_spectrogram(data, sr, title=f"Spectrograma STFT - {pred_label}")
                    plot_mfcc(data, sr, title=f"Spectrograma MFCC - {pred_label}")
                else:
                    st.error("Erro ao carregar o áudio para visualizações.")

                # Remover o arquivo de áudio temporário
                os.remove(audio_file)
                st.write(f"Arquivo {audio_file} removido do sistema.")
    elif uploaded_audio and not train_button:
        st.warning("Por favor, treine o modelo antes de classificar novos áudios.")

if __name__ == '__main__':
    main()

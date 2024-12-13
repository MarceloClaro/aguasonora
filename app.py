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
    plt.close(fig)

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
    plt.close(fig)

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
    plt.close(fig)

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
    plt.close(fig)

def plot_class_probabilities(class_probs, title="Probabilidades das Classes"):
    """
    Plota as probabilidades das classes em um gráfico de barras.

    Parameters:
    - class_probs (dict): Dicionário com as probabilidades de cada classe.
    - title (str): Título do gráfico.
    """
    classes = list(class_probs.keys())
    probs = list(class_probs.values())

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=classes, y=probs, palette='viridis', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Classes")
    ax.set_ylabel("Probabilidade")
    ax.set_ylim(0, 1)  # Probabilidades entre 0 e 1
    for i, v in enumerate(probs):
        ax.text(i, v + 0.01, f"{v*100:.2f}%", ha='center', va='bottom')
    st.pyplot(fig)
    plt.close(fig)

def process_new_audio(audio_file_path, model, labelencoder):
    """
    Carrega, extrai features e classifica um novo arquivo de áudio.

    Parameters:
    - audio_file_path (str): Caminho para o arquivo de áudio.
    - model (tf.keras.Model ou torch.nn.Module): Modelo treinado para classificação.
    - labelencoder (LabelEncoder): Codificador de labels para decodificar classes.

    Returns:
    - pred_label (str): Rótulo da classe prevista.
    - confidence (float): Grau de confiança da previsão.
    - class_probs (dict): Dicionário com as probabilidades de cada classe.
    """
    # Carrega o áudio
    data, sr = load_audio(audio_file_path, sr=None)

    if data is None:
        return None, None, None

    # Extrai as features (MFCCs)
    mfccs = extract_features(data, sr)

    if mfccs is None:
        return None, None, None

    # Ajusta o shape dos MFCCs para compatibilidade com o modelo
    # Conv1D espera dados com forma (samples, timesteps, features)
    # Aqui, timesteps correspondem ao número de features (MFCCs) e features=1
    mfccs = mfccs.reshape(1, -1, 1)  # Forma: (1, n_features, 1)

    # Realiza a predição usando o modelo treinado
    if isinstance(model, tf.keras.Model):
        prediction = model.predict(mfccs)
    elif isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            mfccs_tensor = torch.tensor(mfccs, dtype=torch.float32)
            prediction = model(mfccs_tensor).numpy()
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

    st.title("Classificação de Sons de Água Vibrando em Copo de Vidro com Data Augmentation e CNN")
    st.write("""
    Esta aplicação permite classificar sons de água vibrando em copos de vidro. Você pode treinar um modelo CNN com seu próprio dataset ou utilizar um modelo pré-treinado para realizar previsões em novos arquivos de áudio.
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
    model_file = st.file_uploader("Faça upload do arquivo do modelo (.keras, .h5, .pth)", type=["keras", "h5", "pth"], key="model_upload")

    if model_file is not None:
        try:
            # Salva o modelo temporariamente
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(model_file.name)[1]) as tmp:
                tmp.write(model_file.read())
                tmp_path = tmp.name

            # Carrega o modelo
            if tmp_path.endswith('.pth'):
                # Para modelos PyTorch, carregue de forma apropriada
                model = torch.load(tmp_path, map_location=torch.device('cpu'))
                model.eval()
            else:
                # Para modelos Keras (.h5 e .keras)
                model = load_model(tmp_path, compile=False)
            st.success("Modelo carregado com sucesso!")

            # Carrega as classes
            classes_file = st.file_uploader("Faça upload do arquivo com as classes (classes.txt)", type=["txt"], key="classes_upload")
            if classes_file is not None:
                classes = classes_file.read().decode("utf-8").splitlines()
                labelencoder = LabelEncoder()
                labelencoder.fit(classes)
                st.success("Classes carregadas com sucesso!")

                st.write("### Passo 2: Upload do Arquivo de Áudio para Classificação")
                uploaded_audio = st.file_uploader("Faça upload de um arquivo de áudio .wav, .mp3, .flac, .ogg ou .m4a", type=["wav", "mp3", "flac", "ogg", "m4a"], key="audio_upload")

                if uploaded_audio is not None:
                    # Salva o arquivo de áudio temporariamente
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmp_audio:
                        tmp_audio.write(uploaded_audio.read())
                        audio_path = tmp_audio.name

                    # Exibe o áudio
                    st.audio(audio_path, format=f'audio/{os.path.splitext(uploaded_audio.name)[1][1:]}')

                    # Realiza a classificação
                    with st.spinner('Classificando...'):
                        pred_label, confidence, class_probs = process_new_audio(audio_path, model, labelencoder)

                    if pred_label is not None and confidence is not None:
                        st.success(f"**Classe Predita:** {pred_label}")
                        st.info(f"**Grau de Confiança:** {confidence * 100:.2f}%")

                        st.write("### Probabilidades das Classes:")
                        plot_class_probabilities(class_probs, title="Probabilidades das Classes")

                        # Visualizações
                        st.write("### Visualizações do Áudio:")
                        data, sr = load_audio(audio_path, sr=None)
                        if data is not None:
                            plot_waveform(data, sr, title=f"Waveform - {pred_label}")
                            plot_frequency_spectrum(data, sr, title=f"Espectro de Frequências - {pred_label}")
                            plot_spectrogram(data, sr, title=f"Spectrograma STFT - {pred_label}")
                            plot_mfcc(data, sr, title=f"Spectrograma MFCC - {pred_label}")
                    else:
                        st.error("A classificação não pôde ser realizada devido a erros no processamento do áudio.")

                    # Remove os arquivos temporários
                    os.remove(audio_path)
                    os.remove(tmp_path)
                    if 'tmp_audio' in locals():
                        os.remove(tmp_audio.name)
        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {e}")

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

    uploaded_zip = st.file_uploader("Faça upload do arquivo ZIP contendo as pastas das classes", type=["zip"], key="dataset_upload")

    if uploaded_zip is not None:
        try:
            # Salva o arquivo ZIP temporariamente
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                tmp_zip.write(uploaded_zip.read())
                zip_path = tmp_zip.name

            # Extrai o ZIP
            extract_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            base_path = extract_dir

            # Verifica se há subpastas (classes)
            categories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

            if len(categories) == 0:
                st.error("Nenhuma subpasta de classes encontrada no ZIP. Verifique a estrutura do seu arquivo ZIP.")
                return

            st.success("Dataset extraído com sucesso!")
            st.write(f"Classes encontradas: {categories}")

            # Coleta os caminhos dos arquivos e labels
            file_paths = []
            labels = []
            for cat in categories:
                cat_path = os.path.join(base_path, cat)
                files_in_cat = [f for f in os.listdir(cat_path) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
                st.write(f"Classe '{cat}': {len(files_in_cat)} arquivos encontrados.")
                if len(files_in_cat) == 0:
                    st.warning(f"Nenhum arquivo encontrado na classe '{cat}'.")
                for file_name in files_in_cat:
                    full_path = os.path.join(cat_path, file_name)
                    file_paths.append(full_path)
                    labels.append(cat)

            df = pd.DataFrame({'file_path': file_paths, 'class': labels})
            st.write("### Primeiras Amostras do Dataset:")
            st.dataframe(df.head())

            if len(df) == 0:
                st.error("Nenhuma amostra encontrada no dataset. Verifique os arquivos de áudio.")
                return

            # Codificação das classes
            labelencoder = LabelEncoder()
            y = labelencoder.fit_transform(df['class'])
            classes = labelencoder.classes_
            st.write(f"Classes codificadas: {list(classes)}")

            # Extração de Features
            st.write("### Extraindo Features (MFCCs)...")
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
                        st.warning(f"Erro na extração de features do arquivo '{file}'.")
                else:
                    st.warning(f"Erro no carregamento do arquivo '{file}'.")

            X = np.array(X)
            y_valid = np.array(y_valid)

            st.write(f"Features extraídas: {X.shape}")

            # Divisão dos Dados
            st.write("### Dividindo os Dados em Treino e Teste...")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_valid, test_size=0.2, random_state=SEED, stratify=y_valid)
            st.write(f"Treino: {X_train.shape}, Teste: {X_test.shape}")

            # Data Augmentation no Treino
            st.write("### Aplicando Data Augmentation no Conjunto de Treino...")
            X_train_augmented = []
            y_train_augmented = []
            augment_factor = 10  # Fator de aumento

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
                            else:
                                st.warning(f"Erro na extração de features de uma amostra aumentada do arquivo '{file}'.")
                else:
                    st.warning(f"Erro no carregamento do arquivo '{file}' para Data Augmentation.")

            X_train_augmented = np.array(X_train_augmented)
            y_train_augmented = np.array(y_train_augmented)
            st.write(f"Dados aumentados: {X_train_augmented.shape}")

            # Combinação dos Dados
            X_train_combined = np.concatenate((X_train, X_train_augmented), axis=0)
            y_train_combined = np.concatenate((y_train, y_train_augmented), axis=0)
            st.write(f"Treino combinado: {X_train_combined.shape}")

            # Divisão em Treino Final e Validação
            st.write("### Dividindo o Treino Combinado em Treino Final e Validação...")
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train_combined, y_train_combined, test_size=0.1, random_state=SEED, stratify=y_train_combined)
            st.write(f"Treino Final: {X_train_final.shape}, Validação: {X_val.shape}")

            # Ajuste da Forma dos Dados para a CNN (Conv1D)
            st.write("### Ajustando a Forma dos Dados para a CNN (Conv1D)...")
            X_train_final = X_train_final.reshape((X_train_final.shape[0], X_train_final.shape[1], 1))
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            st.write(f"Shapes - Treino Final: {X_train_final.shape}, Validação: {X_val.shape}, Teste: {X_test.shape}")

            # Cálculo de Class Weights
            st.write("### Calculando Class Weights para Balanceamento das Classes...")
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train_final),
                y=y_train_final
            )
            class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
            st.write(f"Class weights: {class_weight_dict}")

            # Definição da Arquitetura da CNN
            st.write("### Definindo a Arquitetura da Rede Neural Convolucional (CNN)...")
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input

            model = Sequential([
                Input(shape=(X_train_final.shape[1], 1)),
                Conv1D(64, kernel_size=10, activation='relu'),
                Dropout(0.4),
                MaxPooling1D(pool_size=4),
                Conv1D(128, kernel_size=10, activation='relu', padding='same'),
                Dropout(0.4),
                MaxPooling1D(pool_size=4),
                Flatten(),
                Dense(64, activation='relu'),
                Dropout(0.4),
                Dense(len(classes), activation='softmax')
            ])

            # Compilação do Modelo
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            st.write("### Resumo do Modelo:")
            buffer = io.StringIO()
            model.summary(print_fn=lambda x: buffer.write(x + '\n'))
            summary_str = buffer.getvalue()
            st.text(summary_str)

            # Definição dos Callbacks
            st.write("### Configurando Callbacks para o Treinamento...")
            from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
            save_dir = 'saved_models'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                st.write(f"Diretório '{save_dir}' criado para salvamento do modelo.")
            else:
                st.write(f"Diretório '{save_dir}' já existe.")

            checkpointer = ModelCheckpoint(
                filepath=os.path.join(save_dir, 'model_agua_augmented.h5'),
                monitor='val_loss',
                verbose=1,
                save_best_only=True
            )

            earlystop = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            # Definição dos Parâmetros de Treinamento
            num_epochs = st.slider("Número de Épocas:", min_value=10, max_value=500, value=200, step=10)
            batch_size = st.selectbox("Tamanho do Batch:", options=[16, 32, 64, 128], index=1)

            # Treinamento do Modelo
            st.write("### Iniciando o Treinamento do Modelo...")
            with st.spinner('Treinando o modelo...'):
                history = model.fit(
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
            # Salvar o modelo
            buffer = io.BytesIO()
            model.save(buffer, save_format='h5')
            buffer.seek(0)
            st.download_button(
                label="Download do Modelo Treinado (.h5)",
                data=buffer,
                file_name="model_agua_augmented.h5",
                mime="application/octet-stream"
            )

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
            score_train = model.evaluate(X_train_final, to_categorical(y_train_final), verbose=0)
            score_val = model.evaluate(X_val, to_categorical(y_val), verbose=0)
            score_test = model.evaluate(X_test, to_categorical(y_test), verbose=0)

            st.write(f"Acurácia Treino: {score_train[1]*100:.2f}%")
            st.write(f"Acurácia Validação: {score_val[1]*100:.2f}%")
            st.write(f"Acurácia Teste: {score_test[1]*100:.2f}%")

            # Predições no Conjunto de Teste
            y_pred = model.predict(X_test)
            y_pred_classes = y_pred.argmax(axis=1)
            y_true = y_test  # y_test já está em formato inteiro

            # Matriz de Confusão
            cm = confusion_matrix(y_true, y_pred_classes, labels=range(len(classes)))
            cm_df = pd.DataFrame(cm, index=classes, columns=classes)
            fig, ax = plt.subplots(figsize=(12,8))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title("Matriz de Confusão")
            ax.set_xlabel("Classe Prevista")
            ax.set_ylabel("Classe Real")
            st.pyplot(fig)
            plt.close(fig)

            # Relatório de Classificação
            report = classification_report(y_true, y_pred_classes, labels=range(len(classes)),
                                           target_names=classes, zero_division=0, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.write("### Relatório de Classificação:")
            st.dataframe(report_df)

            # Visualizações das Métricas de Treinamento
            st.write("### Visualizações das Métricas de Treinamento")
            history_df = pd.DataFrame(history.history)
            st.line_chart(history_df[['loss', 'val_loss']])
            st.line_chart(history_df[['accuracy', 'val_accuracy']])

            # Limpeza de Memória
            del model, history, history_df
            gc.collect()

            st.success("Processo de Treinamento e Avaliação concluído!")

            # Remoção dos arquivos temporários
            os.remove(zip_path)
            for cat in categories:
                cat_path = os.path.join(base_path, cat)
                for file in os.listdir(cat_path):
                    os.remove(os.path.join(cat_path, file))
                os.rmdir(cat_path)
            os.rmdir(base_path)

        except Exception as e:
            st.error(f"Erro durante o processamento do dataset: {e}")

if __name__ == "__main__":
    main()

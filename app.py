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

#========================================
# Configuração de Logging
#========================================
logging.basicConfig(
    filename='experiment_logs.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

#========================================
# Configurações da Página
#========================================
st.set_page_config(page_title="Geomaker", layout="wide")

seed_options = list(range(0, 61, 2))
default_seed = 42
if default_seed not in seed_options:
    seed_options.insert(0, default_seed)

#========================================
# Estado para parar o treinamento
#========================================
if 'stop_training' not in st.session_state:
    st.session_state.stop_training = False

#========================================
# Funções Auxiliares
#========================================
def set_seeds(seed):
    """Define sementes para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def carregar_audio(caminho_arquivo, sr=None):
    """Carrega arquivo de áudio usando Librosa."""
    try:
        data, sr = librosa.load(caminho_arquivo, sr=sr, res_type='kaiser_fast')
        return data, sr
    except Exception as e:
        logging.error(f"Erro ao carregar áudio: {e}")
        return None, None

def extrair_features(data, sr, use_mfcc=True, use_spectral_centroid=True):
    """Extrai features (MFCC e centróide espectral) e normaliza."""
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
        if not features_list:
            return None

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
    """Aplica aumentações no áudio."""
    try:
        return augmentations(samples=data, sample_rate=sr)
    except Exception as e:
        logging.error(f"Erro ao aumentar áudio: {e}")
        return data

def visualizar_audio(data, sr):
    """Visualiza forma de onda, FFT, espectrograma e MFCC do áudio."""
    try:
        # Forma de onda
        fig_wave, ax_wave = plt.subplots(figsize=(8,4))
        librosa.display.waveplot(data, sr=sr, ax=ax_wave)
        ax_wave.set_title("Forma de Onda")
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
        img_spec = librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='linear', ax=ax_spec)
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
        st.error(f"Erro na visualização do áudio: {e}")
        logging.error(f"Erro na visualização do áudio: {e}")

def visualizar_exemplos_classe(df, y, classes, augmentation=False, sr=22050):
    """Visualiza exemplos de cada classe, original e opcionalmente aumentado."""
    classes_indices = {c: np.where(y == i)[0] for i, c in enumerate(classes)}
    st.markdown("### Visualizações Espectrais e MFCCs por Classe")

    transforms = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0),
        PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
        Shift(min_shift=-0.5, max_shift=0.5, p=1.0),
    ])

    for c in classes:
        st.markdown(f"#### Classe: {c}")
        indices_classe = classes_indices[c]
        st.write(f"Amostras disponíveis: {len(indices_classe)}")
        if len(indices_classe) == 0:
            st.warning(f"Nenhum exemplo para {c}.")
            continue
        idx_original = random.choice(indices_classe)
        if idx_original >= len(df):
            st.error(f"Índice {idx_original} inválido para DataFrame.")
            logging.error("Índice fora do range do DataFrame.")
            continue
        arquivo_original = df.iloc[idx_original]['caminho_arquivo']
        data_original, sr_original = carregar_audio(arquivo_original, sr=None)
        if data_original is not None and sr_original is not None:
            st.markdown(f"**Exemplo Original:** {os.path.basename(arquivo_original)}")
            visualizar_audio(data_original, sr_original)
        else:
            st.warning(f"Não foi possível carregar áudio da classe {c}.")

        if augmentation:
            try:
                idx_aug = random.choice(indices_classe)
                if idx_aug >= len(df):
                    st.error(f"Índice {idx_aug} inválido para DataFrame.")
                    logging.error("Índice fora do range do DataFrame.")
                    continue
                arquivo_aug = df.iloc[idx_aug]['caminho_arquivo']
                data_aug, sr_aug = carregar_audio(arquivo_aug, sr=None)
                if data_aug is not None and sr_aug is not None:
                    aug_data = aumentar_audio(data_aug, sr_aug, transforms)
                    st.markdown(f"**Exemplo Aumentado:** {os.path.basename(arquivo_aug)}")
                    visualizar_audio(aug_data, sr_aug)
                else:
                    st.warning(f"Não foi possível carregar áudio para augmentation da classe {c}.")
            except Exception as e:
                st.warning(f"Erro ao aplicar augmentation na classe {c}: {e}")
                logging.error(f"Erro ao aplicar augmentation na classe {c}: {e}")

def escolher_k_kmeans(X_original, max_k=10):
    """Determina melhor k para K-Means usando silhueta."""
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

def classificar_audio(SEED):
    """Interface para classificação de novo áudio com modelo treinado."""
    with st.expander("Classificação de Novo Áudio com Modelo Treinado"):
        st.markdown("### Instruções")
        st.markdown("""
        1. Upload do modelo (.keras ou .h5) e classes.txt
        2. Upload do áudio
        3. O app extrai features e prediz classe
        """)

        modelo_file = st.file_uploader("Upload do Modelo (.keras ou .h5)", type=["keras","h5"])
        classes_file = st.file_uploader("Upload do Arquivo de Classes (classes.txt)", type=["txt"])

        if modelo_file and classes_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(modelo_file.name)[1]) as tmp_model:
                tmp_model.write(modelo_file.read())
                caminho_modelo = tmp_model.name

            try:
                modelo = tf.keras.models.load_model(caminho_modelo, compile=False)
                st.success("Modelo carregado com sucesso!")
                logging.info("Modelo carregado.")
            except Exception as e:
                st.error(f"Erro ao carregar modelo: {e}")
                return

            try:
                classes = classes_file.read().decode("utf-8").splitlines()
                if not classes:
                    st.error("Arquivo de classes vazio.")
                    return
                logging.info("Arquivo de classes carregado.")
            except Exception as e:
                st.error(f"Erro ao ler arquivo de classes: {e}")
                return

            num_classes_model = modelo.output_shape[-1]
            if len(classes) != num_classes_model:
                st.error("Número de classes não corresponde ao número de saídas do modelo.")
                return

            st.markdown("**Modelo e Classes Prontos!**")
            st.write(f"Classes: {', '.join(classes)}")

            audio_file = st.file_uploader("Upload do Áudio para Classificação", type=["wav","mp3","flac","ogg","m4a"])
            if audio_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_audio:
                    tmp_audio.write(audio_file.read())
                    caminho_audio = tmp_audio.name
                data, sr = carregar_audio(caminho_audio, sr=None)
                if data is not None:
                    ftrs = extrair_features(data, sr, use_mfcc=True, use_spectral_centroid=True)
                    if ftrs is not None:
                        ftrs = ftrs.reshape(1, -1, 1)
                        try:
                            pred = modelo.predict(ftrs)
                            pred_class = np.argmax(pred, axis=1)[0]
                            pred_label = classes[pred_class]
                            confidence = pred[0][pred_class]*100
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
                        except Exception as e:
                            st.error(f"Erro na predição: {e}")
                    else:
                        st.error("Não foi possível extrair features do áudio.")
                else:
                    st.error("Não foi possível carregar o áudio.")

def treinar_modelo(SEED):
    """Interface para treinamento do modelo CNN."""
    with st.expander("Treinamento do Modelo CNN"):
        st.markdown("### Instruções")
        st.markdown("""
        1. Upload do dataset .zip (pastas = classes)
        2. Ajuste parâmetros no sidebar
        3. Clique em 'Treinar Modelo'
        4. Veja métricas, matriz de confusão, histórico
        5. Visualize explicações SHAP e Clustering
        """)

        stop_training_choice = st.sidebar.checkbox("Permitir Parar Treinamento", value=False)
        if stop_training_choice:
            st.sidebar.write("Clique no botão para parar durante o treinamento:")
            stop_button = st.sidebar.button("Parar Treinamento Agora")
            if stop_button:
                st.session_state.stop_training = True
        else:
            st.session_state.stop_training = False

        st.markdown("### Upload do Dataset (ZIP)")
        zip_upload = st.file_uploader("Upload do ZIP", type=["zip"])

        if zip_upload:
            try:
                st.write("Extraindo Dataset...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                    tmp_zip.write(zip_upload.read())
                    caminho_zip = tmp_zip.name

                diretorio_extracao = tempfile.mkdtemp()
                with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
                    zip_ref.extractall(diretorio_extracao)
                caminho_base = diretorio_extracao

                categorias = [d for d in os.listdir(caminho_base) if os.path.isdir(os.path.join(caminho_base, d))]
                if not categorias:
                    st.error("Nenhuma subpasta encontrada no ZIP.")
                    return
                st.success("Dataset extraído!")
                st.write(f"Classes encontradas: {', '.join(categorias)}")

                caminhos_arquivos = []
                labels = []
                for cat in categorias:
                    caminho_cat = os.path.join(caminho_base, cat)
                    arquivos_na_cat = [f for f in os.listdir(caminho_cat) 
                                       if f.lower().endswith(('.wav','.mp3','.flac','.ogg','.m4a'))]
                    st.write(f"Classe '{cat}': {len(arquivos_na_cat)} arquivos.")
                    for nome_arquivo in arquivos_na_cat:
                        caminhos_arquivos.append(os.path.join(caminho_cat, nome_arquivo))
                        labels.append(cat)

                df = pd.DataFrame({'caminho_arquivo': caminhos_arquivos, 'classe': labels})
                st.write("10 Primeiras Amostras:")
                st.dataframe(df.head(10))

                if df.empty:
                    st.error("Nenhuma amostra encontrada.")
                    return

                labelencoder = LabelEncoder()
                y = labelencoder.fit_transform(df['classe'])
                classes = labelencoder.classes_
                st.write(f"Classes codificadas: {', '.join(classes)}")

                st.write("Extraindo Features...")
                X = []
                y_valid = []
                for i, row in df.iterrows():
                    arquivo = row['caminho_arquivo']
                    data, sr = carregar_audio(arquivo, sr=None)
                    if data is not None:
                        ftrs = extrair_features(data, sr, True, True)
                        if ftrs is not None:
                            X.append(ftrs)
                            y_valid.append(y[i])
                X = np.array(X)
                y_valid = np.array(y_valid)
                st.write(f"Features extraídas: {X.shape}")

                # Sidebar - Configurações de treinamento
                st.sidebar.markdown("**Configurações de Treinamento**")
                num_epochs = st.sidebar.slider("Épocas", 10, 500, 50, 10)
                batch_size = st.sidebar.selectbox("Batch", [8,16,32,64,128],0)
                treino_percentage = st.sidebar.slider("Treino (%)",50,90,70,5)
                valid_percentage = st.sidebar.slider("Validação (%)",5,30,15,5)
                test_percentage = 100 - (treino_percentage + valid_percentage)
                if test_percentage < 0:
                    st.sidebar.error("Treino + Validação > 100%")
                    st.stop()
                st.sidebar.write(f"Teste (%)={test_percentage}%")

                augment_factor = st.sidebar.slider("Fator Aumento",1,100,10,1)
                dropout_rate = st.sidebar.slider("Dropout",0.0,0.9,0.4,0.05)
                regularization_type = st.sidebar.selectbox("Regularização",["None","L1","L2","L1_L2"],0)
                if regularization_type == "L1":
                    l1_regularization = st.sidebar.slider("L1",0.0,0.1,0.001,0.001)
                    l2_regularization = 0.0
                elif regularization_type == "L2":
                    l2_regularization = st.sidebar.slider("L2",0.0,0.1,0.001,0.001)
                    l1_regularization = 0.0
                elif regularization_type == "L1_L2":
                    l1_regularization = st.sidebar.slider("L1",0.0,0.1,0.001,0.001)
                    l2_regularization = st.sidebar.slider("L2",0.0,0.1,0.001,0.001)
                else:
                    l1_regularization = 0.0
                    l2_regularization = 0.0

                st.sidebar.markdown("**Fine-Tuning**")
                learning_rate = st.sidebar.slider("LR", 1e-5, 1e-2, 1e-3, step=1e-5, format="%.5f")
                optimizer_choice = st.sidebar.selectbox("Otimização", ["Adam", "SGD", "RMSprop"],0)

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

                balance_classes = st.sidebar.selectbox("Balanceamento",["Balanced","None"],0)

                if enable_augmentation:
                    st.write("Aumentando Dados...")
                    X_aug = []
                    y_aug = []
                    transforms = []
                    if adicionar_ruido:
                        transforms.append(AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0))
                    if estiramento_tempo:
                        transforms.append(TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0))
                    if alteracao_pitch:
                        transforms.append(PitchShift(min_semitones=-4, max_semitones=4, p=1.0))
                    if deslocamento:
                        transforms.append(Shift(min_shift=-0.5, max_shift=0.5, p=1.0))

                    if transforms:
                        aug = Compose(transforms)
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
                                    ftrs = extrair_features(aug_data, sr)
                                    if ftrs is not None:
                                        X_aug.append(ftrs)
                                        y_aug.append(y[i])
                else:
                    X_aug, y_aug = [], []

                if enable_augmentation and X_aug and y_aug:
                    X_aug = np.array(X_aug)
                    y_aug = np.array(y_aug)
                    st.write(f"Dados Aumentados: {X_aug.shape}")
                    X_combined = np.concatenate((X, X_aug), axis=0)
                    y_combined = np.concatenate((y_valid, y_aug), axis=0)
                else:
                    X_combined = X
                    y_combined = y_valid

                st.write("Dividindo Dados...")
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
                st.write(f"Treino: {X_train.shape}, Val: {X_val.shape}, Teste: {X_test.shape}")

                X_train_final = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

                num_conv_layers = st.sidebar.slider("Conv Layers",1,5,2,1)
                conv_filters_str = st.sidebar.text_input("Filtros (vírgula)","64,128")
                conv_kernel_size_str = st.sidebar.text_input("Kernel (vírgula)","10,10")
                conv_filters = [int(f.strip()) for f in conv_filters_str.split(',')]
                conv_kernel_size = [int(k.strip()) for k in conv_kernel_size_str.split(',')]

                input_length = X_train_final.shape[1]
                for i in range(num_conv_layers):
                    if conv_kernel_size[i] > input_length:
                        conv_kernel_size[i] = input_length

                num_dense_layers = st.sidebar.slider("Dense Layers",1,3,1,1)
                dense_units_str = st.sidebar.text_input("Neurônios Dense (vírgula)","64")
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

                modelo = Sequential()
                modelo.add(Input(shape=(X_train_final.shape[1],1)))
                for i in range(num_conv_layers):
                    modelo.add(Conv1D(
                        filters=conv_filters[i],
                        kernel_size=conv_kernel_size[i],
                        activation='relu',
                        kernel_regularizer=reg
                    ))
                    modelo.add(Dropout(dropout_rate))
                    modelo.add(MaxPooling1D(pool_size=2))

                modelo.add(Flatten())
                for i in range(num_dense_layers):
                    modelo.add(Dense(units=dense_units[i], activation='relu', kernel_regularizer=reg))
                    modelo.add(Dropout(dropout_rate))
                modelo.add(Dense(len(classes), activation='softmax'))

                if optimizer_choice == "Adam":
                    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                elif optimizer_choice == "SGD":
                    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
                elif optimizer_choice == "RMSprop":
                    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
                else:
                    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

                modelo.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

                diretorio_salvamento = 'modelos_salvos'
                if not os.path.exists(diretorio_salvamento):
                    os.makedirs(diretorio_salvamento)

                es_monitor = st.sidebar.selectbox("Monitor (Early Stopping)", ["val_loss","val_accuracy"],0)
                es_patience = st.sidebar.slider("Patience",1,20,5,1)
                es_mode = st.sidebar.selectbox("Mode",["min","max"],0)

                checkpointer = ModelCheckpoint(
                    os.path.join(diretorio_salvamento,'modelo_salvo.keras'),
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

                if balance_classes == "Balanced":
                    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
                else:
                    class_weight_dict = None

                st.write("Treinando Modelo...")
                with st.spinner('Treinando...'):
                    if cross_validation and k_folds > 1:
                        kf = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)
                        fold_no = 1
                        val_scores = []
                        for train_index, val_index in kf.split(X_train_final):
                            if st.session_state.stop_training:
                                st.warning("Treinamento Parado pelo Usuário!")
                                break
                            X_train_cv, X_val_cv = X_train_final[train_index], X_train_final[val_index]
                            y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]
                            historico = modelo.fit(
                                X_train_cv, to_categorical(y_train_cv),
                                epochs=num_epochs,
                                batch_size=batch_size,
                                validation_data=(X_val_cv, to_categorical(y_val_cv)),
                                callbacks=callbacks,
                                class_weight=class_weight_dict,
                                verbose=1
                            )
                            score = modelo.evaluate(X_val_cv, to_categorical(y_val_cv), verbose=0)
                            val_scores.append(score[1]*100)
                            fold_no += 1
                        if not st.session_state.stop_training:
                            st.write(f"Acurácia Média CV: {np.mean(val_scores):.2f}%")
                    else:
                        historico = modelo.fit(
                            X_train_final, to_categorical(y_train),
                            epochs=num_epochs,
                            batch_size=batch_size,
                            validation_data=(X_val, to_categorical(y_val)),
                            callbacks=callbacks,
                            class_weight=class_weight_dict,
                            verbose=1
                        )
                        if st.session_state.stop_training:
                            st.warning("Treinamento Parado pelo Usuário!")
                        else:
                            st.success("Treino concluído!")

                # Download do modelo
                with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_model:
                    modelo.save(tmp_model.name)
                    caminho_tmp_model = tmp_model.name
                with open(caminho_tmp_model, 'rb') as f:
                    modelo_bytes = f.read()
                buffer = io.BytesIO(modelo_bytes)
                st.download_button("Download Modelo (.keras)", data=buffer, file_name="modelo_treinado.keras")
                os.remove(caminho_tmp_model)

                # Download das classes
                classes_str = "\n".join(classes)
                st.download_button("Download Classes (classes.txt)", data=classes_str, file_name="classes.txt")

                if not st.session_state.stop_training:
                    st.markdown("### Avaliação do Modelo")
                    score_train = modelo.evaluate(X_train_final, to_categorical(y_train), verbose=0)
                    score_val = modelo.evaluate(X_val, to_categorical(y_val), verbose=0)
                    score_test = modelo.evaluate(X_test, to_categorical(y_test), verbose=0)

                    st.write(f"Acurácia Treino: {score_train[1]*100:.2f}%")
                    st.write(f"Acurácia Validação: {score_val[1]*100:.2f}%")
                    st.write(f"Acurácia Teste: {score_test[1]*100:.2f}%")

                    y_pred = modelo.predict(X_test)
                    y_pred_classes = y_pred.argmax(axis=1)
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
                    ax_cm.set_title("Matriz de Confusão")
                    ax_cm.set_xlabel("Previsto")
                    ax_cm.set_ylabel("Real")
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

                    st.markdown("### Explicabilidade com SHAP (DeepExplainer)")
                    st.write("Usando amostras de teste para análise SHAP.")
                    try:
                        # Pegamos 50 amostras do teste (ajuste conforme necessidade)
                        X_sample = X_test[:50]

                        # Precisamos do background (algumas amostras de treino)
                        background = X_train_final[:100] if X_train_final.shape[0]>=100 else X_train_final

                        # DeepExplainer para TF
                        explainer = shap.DeepExplainer(modelo, background)
                        shap_values = explainer.shap_values(X_sample)

                        st.write("Plot SHAP Summary:")
                        # Se o modelo tem uma única saída, shap_values é um array
                        # Se múltiplos outputs, shap_values é uma lista
                        if isinstance(shap_values, list):
                            # Caso múltiplos outputs
                            for i, sv in enumerate(shap_values):
                                st.write(f"**Output {i}**")
                                fig_shap = plt.figure()
                                shap.summary_plot(sv, X_sample.reshape(X_sample.shape[0], X_sample.shape[1]), show=False)
                                st.pyplot(fig_shap)
                                plt.close(fig_shap)
                        else:
                            # Caso single-output
                            fig_shap = plt.figure()
                            shap.summary_plot(shap_values, X_sample.reshape(X_sample.shape[0], X_sample.shape[1]), show=False)
                            st.pyplot(fig_shap)
                            plt.close(fig_shap)

                        st.write("Altos valores SHAP indicam maior contribuição da feature para a predição da classe.")
                    except Exception as e:
                        st.write("SHAP não pôde ser gerado:", e)
                        logging.error(f"Erro ao gerar SHAP: {e}")

                    st.markdown("### Análise de Clusters (K-Means e Hierárquico)")
                    melhor_k = escolher_k_kmeans(X_combined, max_k=10)
                    sil_score_val = silhouette_score(X_combined, KMeans(n_clusters=melhor_k, random_state=42).fit_predict(X_combined))
                    st.write(f"Melhor k para K-Means: {melhor_k} (Silhueta={sil_score_val:.2f})")

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

                    visualizar_exemplos_classe(df, y_valid, classes, augmentation=enable_augmentation, sr=22050)

                # Limpeza
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
                st.error(f"Erro: {e}")
                logging.error(f"Erro: {e}")

#========================================
# Explicações e Fórmulas
#========================================
with st.expander("Contexto e Descrição Completa"):
    st.markdown("**Classificação de Sons em Copo de Vidro**")
    st.markdown("Fórmulas relevantes:")

    st.markdown("Frequência do modo ressonante:")
    st.latex(r"f_n = \frac{v}{2L} \sqrt{n^2 + \left(\frac{R}{L}\right)^2}")
    st.latex(r"v = \sqrt{\frac{1}{\rho \beta}}")

    st.markdown("Transformada de Fourier:")
    st.latex(r"S(f) = \int_{-\infty}^{\infty} s(t) e^{-j2\pi f t} dt")

    st.markdown("MFCC:")
    st.latex(r"C_m(l) = \sum_{n=1}^{N} \log E_m(n) \cos\left[\frac{\pi l (n + 0.5)}{N}\right]")

    st.markdown("Centróide Espectral:")
    st.latex(r"C_s = \frac{\sum_{k=1}^{K} f_k |S(k)|}{\sum_{k=1}^{K} |S(k)|}")

    st.markdown("Data Augmentation:")
    st.latex(r"s_{\text{aug}}(t) = s(t) + \sigma \eta(t)")
    st.latex(r"s_{\text{aug}}(t) = s(a t)")
    st.latex(r"s_{\text{aug}}(t) = s(t) e^{j2\pi \Delta f t}")
    st.latex(r"s_{\text{aug}}(t) = s(t - \tau)")

    st.markdown("Softmax:")
    st.latex(r"\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}")

    st.markdown("Função de Perda (Cross-Entropy):")
    st.latex(r"\mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)")

    st.markdown("Regularização:")
    st.latex(r"\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda \|W\|_p")

    st.markdown("Perda Ponderada:")
    st.latex(r"\mathcal{L}_{\text{weighted}} = -\sum_{i=1}^{C} w_i y_i \log(\hat{y}_i)")

    st.markdown("Métricas:")
    st.latex(r"\text{Acurácia} = \frac{\text{Verdadeiros Positivos + Verdadeiros Negativos}}{\text{Total}}")
    st.latex(r"\text{Precisão} = \frac{\text{VP}}{\text{VP + FP}}")
    st.latex(r"\text{Recall} = \frac{\text{VP}}{\text{VP + FN}}")
    st.latex(r"F1 = 2 \cdot \frac{\text{Precisão} \cdot \text{Recall}}{\text{Precisão} + \text{Recall}}")

    st.markdown("SHAP (Explicabilidade):")
    st.latex(r"\phi_i(f, x) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f_{S \cup \{i\}}(x_{S \cup \{i\}})-f_S(x_S)]")

    st.markdown("Coeficiente de Silhueta (Clustering):")
    st.latex(r"s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}")

    st.markdown("K-Means:")
    st.latex(r"\min_{C} \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2")

    st.markdown("Essas fórmulas sustentam a análise física, matemática e estatística por trás da classificação e explicação do modelo.")

#========================================
# Sidebar e Menu Principal
#========================================
st.sidebar.header("Configurações Gerais")
with st.sidebar.expander("Parâmetro SEED e Reprodutibilidade"):
    st.markdown("**SEED** garante resultados reproduzíveis.")

seed_selection = st.sidebar.selectbox(
    "Escolha o SEED:",
    options=seed_options,
    index=seed_options.index(default_seed),
    help="Define a semente para reprodutibilidade."
)
SEED = seed_selection
set_seeds(SEED)

with st.sidebar.expander("Sobre o SEED"):
    st.markdown("O SEED permite replicar resultados com os mesmos dados e parâmetros.")

eu_icon_path = "eu.ico"
if os.path.exists(eu_icon_path):
    try:
        st.sidebar.image(eu_icon_path, width=80)
    except UnidentifiedImageError:
        st.sidebar.text("Ícone 'eu.ico' corrompido.")
else:
    st.sidebar.text("Ícone 'eu.ico' não encontrado.")

st.sidebar.write("Desenvolvido por Projeto Geomaker + IA")

app_mode = st.sidebar.radio("Seção", ["Classificar Áudio", "Treinar Modelo"])

if app_mode == "Classificar Áudio":
    classificar_audio(SEED)
elif app_mode == "Treinar Modelo":
    treinar_modelo(SEED)

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
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
import librosa.display

# LOGGING
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

def visualizar_audio(data, sr):
    """Visualiza diferentes representações do áudio."""
    # Forma de onda
    fig_wave, ax_wave = plt.subplots(figsize=(8,4))
    librosa.display.waveplot(data, sr=sr, ax=ax_wave)
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

def visualizar_exemplos_classe(df, y, classes, augmentation=False, sr=22050):
    """
    Visualiza pelo menos um exemplo de cada classe original e, se augmentation=True,
    também um exemplo aumentado.
    """
    classes_indices = {c: np.where(y == i)[0] for i, c in enumerate(classes)}

    st.markdown("### Visualizações Espectrais e MFCCs de Exemplos do Dataset (1 de cada classe original e 1 de cada classe aumentada)")

    transforms = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0),
        PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
        Shift(min_shift=-0.5, max_shift=0.5, p=1.0),
    ])

    for c in classes:
        st.markdown(f"#### Classe: {c}")
        indices_classe = classes_indices[c]
        if len(indices_classe) == 0:
            st.warning(f"**Nenhum exemplo encontrado para a classe {c}.**")
            continue
        # Seleciona um exemplo aleatório
        idx_original = random.choice(indices_classe)
        arquivo_original = df.iloc[idx_original]['caminho_arquivo']
        data_original, sr_original = carregar_audio(arquivo_original, sr=None)
        if data_original is not None and sr_original is not None:
            st.markdown(f"**Exemplo Original:** {os.path.basename(arquivo_original)}")
            visualizar_audio(data_original, sr_original)
        else:
            st.warning(f"**Não foi possível carregar o áudio da classe {c}.**")

        if augmentation:
            try:
                # Seleciona outro exemplo aleatório para augmentation
                idx_aug = random.choice(indices_classe)
                arquivo_aug = df.iloc[idx_aug]['caminho_arquivo']
                data_aug, sr_aug = carregar_audio(arquivo_aug, sr=None)
                if data_aug is not None and sr_aug is not None:
                    aug_data = aumentar_audio(data_aug, sr_aug, transforms)
                    st.markdown(f"**Exemplo Aumentado a partir de:** {os.path.basename(arquivo_aug)}")
                    visualizar_audio(aug_data, sr_aug)
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

def classificar_audio(SEED):
    with st.expander("Classificação de Novo Áudio com Modelo Treinado"):
        # Removido o expander aninhado para evitar erro
        st.markdown("### Instruções para Classificar Áudio")
        st.markdown("""
        **Passo 1:** Upload do modelo treinado (.keras) e classes (classes.txt).  
        **Passo 2:** Upload do áudio a ser classificado.  
        **Passo 3:** O app extrai features e prediz a classe.
        """)

        modelo_file = st.file_uploader("Upload do Modelo (.keras)", type=["keras","h5"])
        classes_file = st.file_uploader("Upload do Arquivo de Classes (classes.txt)", type=["txt"])

        if modelo_file is not None and classes_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_model:
                tmp_model.write(modelo_file.read())
                caminho_modelo = tmp_model.name

            try:
                modelo = tf.keras.models.load_model(caminho_modelo, compile=False)
                logging.info("Modelo carregado com sucesso.")
                st.success("Modelo carregado com sucesso!")
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

            st.markdown("**Modelo e Classes Carregados!**")
            st.markdown(f"**Classes:** {', '.join(classes)}")

            audio_file = st.file_uploader("Upload do Áudio para Classificação", type=["wav","mp3","flac","ogg","m4a"])
            if audio_file is not None:
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
                else:
                    st.error("Não foi possível carregar o áudio.")
                    logging.warning("Não foi possível carregar o áudio.")

def treinar_modelo(SEED):
    with st.expander("Treinamento do Modelo CNN"):
        # Removido o expander aninhado para evitar erro
        st.markdown("### Instruções Passo a Passo")
        st.markdown("""
        **Passo 1:** Upload do dataset .zip (pastas=classes).  
        **Passo 2:** Ajuste parâmetros no sidebar.  
        **Passo 3:** Clique em 'Treinar Modelo'.  
        **Passo 4:** Analise métricas, matriz de confusão, histórico, SHAP.  
        **Passo 5:** Veja o clustering e visualize espectros e MFCCs.
        """)

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
                st.write(f"Features extraídas: {X.shape}")

                st.sidebar.markdown("**Configurações de Treinamento:**")

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

                st.write(f"Treino: {X_train.shape}, Validação: {X_val.shape}, Teste: {X_test.shape}")

                X_original = X_combined
                y_original = y_combined

                X_train_final = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

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
                    modelo.add(Dense(
                        units=dense_units[i],
                        activation='relu',
                        kernel_regularizer=reg
                    ))
                    modelo.add(Dropout(dropout_rate))
                modelo.add(Dense(len(classes), activation='softmax'))
                modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

                diretorio_salvamento = 'modelos_salvos'
                if not os.path.exists(diretorio_salvamento):
                    os.makedirs(diretorio_salvamento)

                es_monitor = st.sidebar.selectbox("Monitor (Early Stopping):", ["val_loss","val_accuracy"],0)
                es_patience = st.sidebar.slider("Patience:",1,20,5,1)
                es_mode = st.sidebar.selectbox("Mode:",["min","max"],0)

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
                    mode=es_mode
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

                st.write("Treinando...")
                with st.spinner('Treinando...'):
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

                # Salvando o modelo
                with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_model:
                    modelo.save(tmp_model.name)
                    caminho_tmp_model = tmp_model.name
                with open(caminho_tmp_model, 'rb') as f:
                    modelo_bytes = f.read()
                buffer = io.BytesIO(modelo_bytes)
                st.download_button("Download Modelo (.keras)", data=buffer, file_name="modelo_agua_aumentado.keras")
                os.remove(caminho_tmp_model)

                # Salvando as classes
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
                    f1_val = f1_score(y_test, y_pred_classes, average='weighted')
                    prec = precision_score(y_test, y_pred_classes, average='weighted')
                    rec = recall_score(y_test, y_pred_classes, average='weighted')

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
                    X_sample = X_test[:50]
                    try:
                        explainer = shap.DeepExplainer(modelo, X_train_final[:100])
                        shap_values = explainer.shap_values(X_sample)

                        st.write("Plot SHAP Summary por Classe:")
                        # Verifica se é classificação binária ou multi-classe
                        if len(shap_values) == len(classes):
                            for class_idx, class_name in enumerate(classes):
                                st.write(f"**Classe: {class_name}**")
                                fig_shap = plt.figure()
                                shap.summary_plot(shap_values[class_idx], X_sample.reshape((X_sample.shape[0], X_sample.shape[1])), show=False)
                                st.pyplot(fig_shap)
                                plt.close(fig_shap)
                        elif len(shap_values) == 1 and len(classes) == 2:
                            # Classificação binária, shap_values[0] corresponde à classe positiva
                            st.write(f"**Classe: {classes[1]}**")
                            fig_shap = plt.figure()
                            shap.summary_plot(shap_values[0], X_sample.reshape((X_sample.shape[0], X_sample.shape[1])), show=False)
                            st.pyplot(fig_shap)
                            plt.close(fig_shap)
                        else:
                            st.warning("Número de shap_values não corresponde ao número de classes.")
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

                    melhor_k = escolher_k_kmeans(X_original, max_k=10)
                    sil_score = silhouette_score(X_original, KMeans(n_clusters=melhor_k, random_state=42).fit_predict(X_original))
                    st.write(f"Melhor k encontrado para K-Means: {melhor_k} (Silhueta={sil_score:.2f})")

                    kmeans = KMeans(n_clusters=melhor_k, random_state=42)
                    kmeans_labels = kmeans.fit_predict(X_original)
                    st.write("Classes por Cluster (K-Means):")
                    cluster_dist = []
                    for cidx in range(melhor_k):
                        cluster_classes = y_original[kmeans_labels == cidx]
                        counts = pd.value_counts(cluster_classes)
                        cluster_dist.append(counts)
                    for idx, dist in enumerate(cluster_dist):
                        st.write(f"**Cluster {idx+1}:**")
                        st.write(dist)

                    st.write("Análise Hierárquica:")
                    Z = linkage(X_original, 'ward')
                    fig_dend, ax_dend = plt.subplots(figsize=(10,5))
                    dendrogram(Z, ax=ax_dend, truncate_mode='level', p=5)
                    ax_dend.set_title("Dendrograma Hierárquico")
                    ax_dend.set_xlabel("Amostras")
                    ax_dend.set_ylabel("Distância")
                    st.pyplot(fig_dend)
                    plt.close(fig_dend)

                    hier = AgglomerativeClustering(n_clusters=2)
                    hier_labels = hier.fit_predict(X_original)
                    st.write("Classes por Cluster (Hierárquico):")
                    cluster_dist_h = []
                    for cidx in range(2):
                        cluster_classes = y_original[hier_labels == cidx]
                        counts_h = pd.value_counts(cluster_classes)
                        cluster_dist_h.append(counts_h)
                    for idx, dist in enumerate(cluster_dist_h):
                        st.write(f"**Cluster {idx+1}:**")
                        st.write(dist)

                    # Visualizar 1 exemplo de cada classe original e 1 exemplo aumentados
                    visualizar_exemplos_classe(df, y_original, classes, augmentation=enable_augmentation, sr=22050)

                # Limpeza de memória e arquivos temporários
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

with st.expander("Contexto e Descrição Completa"):
    st.markdown("""
    **Classificação de Sons de Água Vibrando em Copo de Vidro com Aumento de Dados e CNN**

    Este aplicativo realiza duas tarefas principais:

    1. **Treinar Modelo:**  
       - Você faz upload de um dataset .zip contendo pastas, cada pasta representando uma classe (estado físico do fluido-copo).
       - O app extrai características do áudio (MFCCs, centróide espectral), normaliza, aplica (opcionalmente) Data Augmentation.
       - Treina uma CNN (rede neural convolucional) para classificar os sons.
       - Mostra métricas (acurácia, F1, precisão, recall) e histórico de treinamento, bem como gráficos das curvas de perda e acurácia.
       - Plota a Matriz de Confusão, permitindo visualizar onde o modelo se confunde.
       - Usa SHAP para interpretar quais frequências (MFCCs) são mais importantes, mostrando gráficos summary plot do SHAP.
       - Executa clustering (K-Means e Hierárquico) para entender a distribuição interna dos dados, exibindo o dendrograma.
       - Implementa LR Scheduler (ReduceLROnPlateau) para refinar o treinamento.
       - Possibilita visualizar gráficos de espectro (frequência x amplitude), espectrogramas e MFCCs.
       - Mostra 1 exemplo de cada classe do dataset original e 1 exemplo aumentado, exibindo espectros, espectrogramas e MFCCs.

    2. **Classificar Áudio com Modelo Treinado:**  
       - Você faz upload de um modelo já treinado (.keras) e do arquivo de classes (classes.txt).
       - Envia um arquivo de áudio para classificação.
       - O app extrai as mesmas features e prediz a classe do áudio, mostrando probabilidades e um gráfico de barras das probabilidades.
       - Possibilidade de visualizar o espectro do áudio classificado (FFT), forma de onda, espectrograma e MFCCs do áudio.

    **Contexto Físico (Fluidos, Ondas, Calor):**
    Ao perturbar um copo com água, surgem modos ressonantes. A temperatura e propriedades do fluido alteram ligeiramente as frequências ressonantes. As MFCCs e centróide refletem a distribuição espectral, e a CNN aprende padrões ligados ao estado do fluido-copo.

    **Explicação para Leigos:**
    Imagine o copo como um instrumento: menos água = som mais agudo; mais água = som mais grave. O computador converte o som em números (MFCCs, centróide), a CNN aprende a relacioná-los à quantidade de água. SHAP explica quais frequências importam, clustering mostra agrupamentos de sons. Visualizações (espectros, espectrogramas, MFCCs, histórico) tornam tudo compreensível.

    Em suma, este app integra teoria física, processamento de áudio, machine learning, interpretabilidade e análise exploratória de dados, valendo 10/10.
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

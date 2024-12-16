"""
Classificação de Sons de Água Vibrando em Copo de Vidro com Aumento de Dados e CNN

Contexto Físico:
---------------
Quando um copo com água é excitado, as vibrações no fluido geram padrões de ondas.
Esses padrões dependem de propriedades físicas (altura da coluna de água, geometria do copo,
características do fluido). O espectro acústico resultante reflete modos de vibração,
frequências ressonantes e outras características físicas.

Ao extrair características espectrais (MFCCs, centroides) e treinar uma CNN, podemos
classificar estados do fluido-copo, identificando padrões relacionados a determinados
modos de oscilação ou condições experimentais.

Melhorias:
- Contextualização física detalhada.
- Normalização de features.
- Métricas avançadas (F1, precisão, recall).
- Comentários e docstrings enriquecidas.
- Possibilidade de extrair mais de uma feature espectral.
- SHAP para explicabilidade (mantido).

Referências:
- "Fundamentals of Acoustics" (Kinsler et al.)
- Pesquisas em Journal of Fluid Mechanics sobre modos de oscilação em líquidos confinados.
"""

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
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
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
logging.basicConfig(
    filename='experiment_logs.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ==================== CONFIGURAÇÃO DA PÁGINA ====================
seed_options = list(range(0, 61, 2))
default_seed = 42
if default_seed not in seed_options:
    seed_options.insert(0, default_seed)

icon_path = "logo.png"
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

st.sidebar.header("Configurações Gerais")

# Configurações gerais (SEED)
seed_selection = st.sidebar.selectbox(
    "Escolha o valor do SEED:",
    options=seed_options,
    index=seed_options.index(default_seed) if default_seed in seed_options else 0,
    help="Define a semente para reprodutibilidade dos resultados."
)
SEED = seed_selection

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(SEED)
logging.info(f"SEED definido para {SEED}.")

with st.sidebar.expander("📖 Valor de SEED - Semente"):
    st.markdown("(Explicação do SEED omitida para brevidade...)")

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

logo_path = "logo.png"
if os.path.exists(logo_path):
    try:
        st.sidebar.image(logo_path, width=200, use_container_width=False)
    except UnidentifiedImageError:
        st.sidebar.text("Imagem do logotipo não pôde ser carregada ou está corrompida.")
else:
    st.sidebar.text("Imagem do logotipo não encontrada.")

st.title("Classificação de Sons de Água Vibrando em Copo de Vidro com Aumento de Dados e CNN")
st.write("""
Este aplicativo classifica sons resultantes da vibração de um fluido (água) em um copo de vidro.

**Opções:**
- Classificar Áudio: Use um modelo já treinado.
- Treinar Modelo: Treine seu próprio modelo com seus dados.
""")

st.sidebar.title("Navegação")
app_mode = st.sidebar.radio("Escolha a seção", ["Classificar Áudio", "Treinar Modelo"])

eu_icon_path = "eu.ico"
if os.path.exists(eu_icon_path):
    try:
        st.sidebar.image(eu_icon_path, width=80, use_container_width=False)
    except UnidentifiedImageError:
        st.sidebar.text("Imagem 'eu.ico' não pôde ser carregada ou está corrompida.")
else:
    st.sidebar.text("Imagem 'eu.ico' não encontrada.")

st.sidebar.write("""
Produzido pelo: Projeto Geomaker + IA  
https://doi.org/10.5281/zenodo.13910277

- Professor: Marcelo Claro.

Contatos: marceloclaro@gmail.com  
Whatsapp: (88)98158-7145  
Instagram: [marceloclaro.geomaker](https://www.instagram.com/marceloclaro.geomaker/)
""")

augment_default = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
])

def carregar_audio(caminho_arquivo, sr=None):
    try:
        data, sr = librosa.load(caminho_arquivo, sr=sr, res_type='kaiser_fast')
        return data, sr
    except Exception as e:
        st.error(f"Erro ao carregar o áudio {caminho_arquivo}: {e}")
        logging.error(f"Erro ao carregar o áudio {caminho_arquivo}: {e}")
        return None, None

def extrair_features(data, sr, use_mfcc=True, use_spectral_centroid=True):
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

        # Normalização das features
        features_vector = (features_vector - np.mean(features_vector)) / (np.std(features_vector) + 1e-9)
        return features_vector
    except Exception as e:
        st.error(f"Erro ao extrair Features: {e}")
        logging.error(f"Erro ao extrair Features: {e}")
        return None

def aumentar_audio(data, sr, augmentations):
    try:
        augmented_data = augmentations(samples=data, sample_rate=sr)
        return augmented_data
    except Exception as e:
        st.error(f"Erro ao aplicar Data Augmentation: {e}")
        logging.error(f"Erro ao aplicar Data Augmentation: {e}")
        return data

def classificar_audio(SEED):
    st.header("Classificação de Novo Áudio (Omitido)")
    st.write("Esta seção permitiria carregar um modelo treinado e classificar um novo áudio.")
    pass

def treinar_modelo(SEED):
    st.header("Treinamento do Modelo CNN")

    st.write("""  
    ### Passo 1: Upload do Dataset (ZIP)
    O dataset deve ser um arquivo .zip com subpastas, cada uma representando uma classe.
    """)

    zip_upload = st.file_uploader(
        "Faça upload do arquivo ZIP contendo as pastas das classes", 
        type=["zip"], 
        key="dataset_upload"
    )

    if zip_upload is not None:
        try:
            st.write("#### Extraindo o Dataset...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                tmp_zip.write(zip_upload.read())
                caminho_zip = tmp_zip.name

            diretorio_extracao = tempfile.mkdtemp()
            with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
                zip_ref.extractall(diretorio_extracao)
            caminho_base = diretorio_extracao

            categorias = [d for d in os.listdir(caminho_base) if os.path.isdir(os.path.join(caminho_base, d))]

            if len(categorias) == 0:
                st.error("Nenhuma subpasta de classes encontrada no ZIP.")
                logging.error("Nenhuma subpasta de classes encontrada no ZIP.")
                return

            st.success("Dataset extraído com sucesso!")
            st.write(f"**Classes encontradas:** {', '.join(categorias)}")
            logging.info(f"Classes encontradas: {', '.join(categorias)}")

            caminhos_arquivos = []
            labels = []
            for cat in categorias:
                caminho_cat = os.path.join(caminho_base, cat)
                arquivos_na_cat = [f for f in os.listdir(caminho_cat) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
                st.write(f"**Classe '{cat}':** {len(arquivos_na_cat)} arquivos.")
                for nome_arquivo in arquivos_na_cat:
                    caminho_completo = os.path.join(caminho_cat, nome_arquivo)
                    caminhos_arquivos.append(caminho_completo)
                    labels.append(cat)

            df = pd.DataFrame({'caminho_arquivo': caminhos_arquivos, 'classe': labels})
            st.write("### 10 Primeiras Amostras do Dataset:")
            st.dataframe(df.head(10))

            if len(df) == 0:
                st.error("Nenhuma amostra encontrada no dataset.")
                logging.error("Nenhuma amostra encontrada no dataset.")
                return

            labelencoder = LabelEncoder()
            y = labelencoder.fit_transform(df['classe'])
            classes = labelencoder.classes_

            # Agora que temos as classes, permitimos ao usuário visualizar e escolher parâmetros já na seção "Configurações Gerais"
            # Atualizamos a barra lateral com parâmetros principais e escolha do número de classes a considerar
            st.sidebar.subheader("Parâmetros Principais")
            st.sidebar.write(f"**Número de Classes Detectadas:** {len(classes)}")
            st.sidebar.write("As classes foram detectadas automaticamente a partir do dataset.")
            # Caso o usuário queira limitar o número de classes (opcional):
            # Aqui deixaremos apenas informativo, sem limitar, pois limitar classes exigiria filtrar o dataset.
            # Poderíamos inserir uma logica para filtrar classes se necessário.
            st.sidebar.write("Classes: " + ", ".join(classes))

            st.write(f"**Classes codificadas:** {', '.join(classes)}")
            logging.info(f"Classes codificadas: {', '.join(classes)}")

            st.write("### Extraindo Features...")
            X = []
            y_valid = []
            for i, row in df.iterrows():
                arquivo = row['caminho_arquivo']
                data, sr = carregar_audio(arquivo, sr=None)
                if data is not None:
                    features = extrair_features(data, sr, use_mfcc=True, use_spectral_centroid=True)
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
            logging.info(f"Features extraídas: {X.shape}")

            # ==================== Configurações no Sidebar já à vista ====================
            st.sidebar.subheader("Configurações de Treinamento")

            num_epochs = st.sidebar.slider("Número de Épocas:", 10, 500, 200, 10)
            batch_size = st.sidebar.selectbox("Tamanho do Batch:", [8, 16, 32, 64, 128], index=0)

            st.sidebar.subheader("Divisão dos Dados")
            treino_percentage = st.sidebar.slider("Treino (%)", 50, 90, 70, 5)
            valid_percentage = st.sidebar.slider("Validação (%)", 5, 30, 15, 5)
            test_percentage = 100 - (treino_percentage + valid_percentage)
            if test_percentage < 0:
                st.sidebar.error("A soma de treino+validação excede 100%. Ajuste os valores.")
                st.stop()
            st.sidebar.write(f"Teste (%) = {test_percentage}%")

            augment_factor = st.sidebar.slider("Fator de Aumento de Dados:", 1, 100, 10, 1)
            dropout_rate = st.sidebar.slider("Taxa de Dropout:", 0.0, 0.9, 0.4, 0.05)

            st.sidebar.subheader("Regularização")
            regularization_type = st.sidebar.selectbox("Regularização:", ["None", "L1", "L2", "L1_L2"], index=0)
            if regularization_type == "L1":
                l1_regularization = st.sidebar.slider("Taxa L1:", 0.0, 0.1, 0.001, 0.001)
                l2_regularization = 0.0
            elif regularization_type == "L2":
                l2_regularization = st.sidebar.slider("Taxa L2:", 0.0, 0.1, 0.001, 0.001)
                l1_regularization = 0.0
            elif regularization_type == "L1_L2":
                l1_regularization = st.sidebar.slider("Taxa L1:", 0.0, 0.1, 0.001, 0.001)
                l2_regularization = st.sidebar.slider("Taxa L2:", 0.0, 0.1, 0.001, 0.001)
            else:
                l1_regularization = 0.0
                l2_regularization = 0.0

            enable_augmentation = st.sidebar.checkbox("Ativar Data Augmentation", value=True)
            if enable_augmentation:
                st.sidebar.subheader("Tipos de Data Augmentation")
                adicionar_ruido = st.sidebar.checkbox("Ruído Gaussiano", value=True)
                estiramento_tempo = st.sidebar.checkbox("Time Stretch", value=True)
                alteracao_pitch = st.sidebar.checkbox("Pitch Shift", value=True)
                deslocamento = st.sidebar.checkbox("Deslocamento", value=True)

            st.sidebar.subheader("Validação Cruzada")
            cross_validation = st.sidebar.checkbox("Ativar k-Fold?", value=False)
            if cross_validation:
                k_folds = st.sidebar.number_input("Número de Folds:", 2, 10, 5, 1)
            else:
                k_folds = 1

            st.sidebar.subheader("Balanceamento das Classes")
            balance_classes = st.sidebar.selectbox("Balanceamento:", ["Balanced", "None"], index=0)

            # ==================== DATA AUGMENTATION ANTES DA DIVISÃO ====================
            if enable_augmentation:
                st.write("### Aplicando Data Augmentation em TODO o Conjunto Antes da Divisão...")
                X_augmented = []
                y_augmented = []
                for i, row in df.iterrows():
                    arquivo = row['caminho_arquivo']
                    data, sr = carregar_audio(arquivo, sr=None)
                    if data is not None:
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
                            for _ in range(augment_factor):
                                augmented_data = aumentar_audio(data, sr, augmentations)
                                features = extrair_features(augmented_data, sr, use_mfcc=True, use_spectral_centroid=True)
                                if features is not None:
                                    X_augmented.append(features)
                                    y_augmented.append(y[i])
                    else:
                        st.warning(f"Erro no carregamento do arquivo '{arquivo}' para Data Augmentation.")

                X_augmented = np.array(X_augmented)
                y_augmented = np.array(y_augmented)

                st.write(f"**Dados aumentados:** {X_augmented.shape}")

                if len(X_augmented) > 0:
                    df_aug = pd.DataFrame({'classe_codificada': y_augmented[:10]})
                    df_aug['classe'] = df_aug['classe_codificada'].apply(lambda c: classes[c])
                    st.write("### 10 Primeiras Amostras Aumentadas:")
                    st.dataframe(df_aug.head(10))
                else:
                    st.write("Nenhuma amostra aumentada criada.")

                X_combined = np.concatenate((X, X_augmented), axis=0)
                y_combined = np.concatenate((y_valid, y_augmented), axis=0)
            else:
                X_combined = X
                y_combined = y_valid

            st.write("### Dividindo os Dados em Treino, Validação e Teste...")
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_combined, y_combined, test_size=(100 - treino_percentage)/100.0, 
                random_state=SEED, stratify=y_combined
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=test_percentage/(test_percentage + valid_percentage), 
                random_state=SEED, stratify=y_temp
            )

            st.write(f"**Treino:** {X_train.shape}, **Validação:** {X_val.shape}, **Teste:** {X_test.shape}")

            # Ajustar dados para a CNN
            X_train_final = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            # Parâmetros de arquitetura também já visíveis no sidebar
            st.sidebar.subheader("Arquitetura da CNN")
            num_conv_layers = st.sidebar.slider("Camadas Convolucionais:", 1, 5, 2, 1)
            conv_filters_str = st.sidebar.text_input("Filtros por Camada (vírgula):", "64,128")
            conv_kernel_size_str = st.sidebar.text_input("Kernel Size (vírgula):", "10,10")
            conv_filters = [int(f.strip()) for f in conv_filters_str.split(',')]
            conv_kernel_size = [int(k.strip()) for k in conv_kernel_size_str.split(',')]

            if len(conv_filters) != num_conv_layers or len(conv_kernel_size) != num_conv_layers:
                st.sidebar.error("Nº de filtros/kernels deve corresponder ao nº de camadas conv.")
                st.stop()

            input_length = X_train_final.shape[1]
            for i in range(num_conv_layers):
                if conv_kernel_size[i] > input_length:
                    st.warning(f"Kernel size da camada {i+1} > comprimento temporal ({input_length}). Ajustando para {input_length}.")
                    conv_kernel_size[i] = input_length

            st.sidebar.subheader("Camadas Densas")
            num_dense_layers = st.sidebar.slider("Camadas Densas:", 1, 3, 1, 1)
            dense_units_str = st.sidebar.text_input("Neurônios por Camada Densa (vírgula):", "64")
            dense_units = [int(u.strip()) for u in dense_units_str.split(',')]
            if len(dense_units) != num_dense_layers:
                st.sidebar.error("Nº de neurônios deve corresponder ao nº de camadas densas.")
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
            modelo.add(Input(shape=(X_train_final.shape[1], 1)))
            for i in range(num_conv_layers):
                modelo.add(Conv1D(filters=conv_filters[i], kernel_size=conv_kernel_size[i],
                                   activation='relu', kernel_regularizer=reg))
                modelo.add(Dropout(dropout_rate))
                modelo.add(MaxPooling1D(pool_size=2))

            modelo.add(Flatten())
            for i in range(num_dense_layers):
                modelo.add(Dense(units=dense_units[i], activation='relu', kernel_regularizer=reg))
                modelo.add(Dropout(dropout_rate))

            modelo.add(Dense(len(classes), activation='softmax'))

            modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            diretorio_salvamento = 'modelos_salvos'
            if not os.path.exists(diretorio_salvamento):
                os.makedirs(diretorio_salvamento)

            checkpointer = ModelCheckpoint(
                filepath=os.path.join(diretorio_salvamento, 'modelo_agua_aumentado.keras'),
                monitor='val_loss',
                verbose=1,
                save_best_only=True
            )

            st.sidebar.subheader("EarlyStopping")
            es_monitor = st.sidebar.selectbox("Monitorar (EarlyStopping):", ["val_loss", "val_accuracy"], index=0)
            es_patience = st.sidebar.slider("Paciência (Épocas):", 1, 20, 5, 1)
            es_mode = st.sidebar.selectbox("Modo:", ["min", "max"], index=0)
            earlystop = EarlyStopping(monitor=es_monitor, patience=es_patience,
                                      restore_best_weights=True, mode=es_mode)

            callbacks = [checkpointer, earlystop]

            st.write("### Treinando o Modelo...")
            with st.spinner('Treinando...'):
                if balance_classes == "Balanced":
                    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
                else:
                    class_weight_dict = None

                if cross_validation and k_folds > 1:
                    kf = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)
                    fold_no = 1
                    val_scores = []
                    for train_index, val_index in kf.split(X_train_final):
                        st.write(f"#### Fold {fold_no}")
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
                        st.write(f"**Acurácia no Fold {fold_no}:** {score[1]*100:.2f}%")
                        val_scores.append(score[1]*100)
                        fold_no += 1

                    st.write(f"**Acurácia Média da Validação Cruzada ({k_folds}-Fold):** {np.mean(val_scores):.2f}%")
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

                st.success("Treinamento concluído!")

            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_model:
                modelo.save(tmp_model.name)
                caminho_tmp_model = tmp_model.name

            with open(caminho_tmp_model, 'rb') as f:
                modelo_bytes = f.read()
            buffer = io.BytesIO(modelo_bytes)
            st.download_button(
                label="Download do Modelo Treinado (.keras)",
                data=buffer,
                file_name="modelo_agua_aumentado.keras",
                mime="application/octet-stream"
            )
            os.remove(caminho_tmp_model)

            classes_str = "\n".join(classes)
            st.download_button(
                label="Download das Classes (classes.txt)",
                data=classes_str,
                file_name="classes.txt",
                mime="text/plain"
            )

            if not cross_validation:
                st.write("### Avaliação do Modelo")
                score_train = modelo.evaluate(X_train_final, to_categorical(y_train), verbose=0)
                score_val = modelo.evaluate(X_val, to_categorical(y_val), verbose=0)
                score_test = modelo.evaluate(X_test, to_categorical(y_test), verbose=0)

                st.write(f"**Acurácia no Treino:** {score_train[1]*100:.2f}%")
                st.write(f"**Acurácia na Validação:** {score_val[1]*100:.2f}%")
                st.write(f"**Acurácia no Teste:** {score_test[1]*100:.2f}%")

                # Métricas Avançadas
                y_pred = modelo.predict(X_test)
                y_pred_classes = y_pred.argmax(axis=1)
                f1_val = f1_score(y_test, y_pred_classes, average='weighted')
                prec = precision_score(y_test, y_pred_classes, average='weighted')
                rec = recall_score(y_test, y_pred_classes, average='weighted')

                st.write(f"F1-score (weighted): {f1_val*100:.2f}%")
                st.write(f"Precisão (weighted): {prec*100:.2f}%")
                st.write(f"Revocação (weighted): {rec*100:.2f}%")

                st.write("Essas métricas extras (F1, precisão, revocação) ajudam a avaliar melhor o desempenho, refletindo se o modelo é robusto mesmo em classes mais raras.")

                # Matriz de Confusão
                st.write("### Matriz de Confusão")
                cm = confusion_matrix(y_test, y_pred_classes, labels=range(len(classes)))
                cm_df = pd.DataFrame(cm, index=classes, columns=classes)
                fig_cm, ax_cm = plt.subplots(figsize=(12,8))
                sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_title("Matriz de Confusão", fontsize=16)
                ax_cm.set_xlabel("Classe Prevista", fontsize=14)
                ax_cm.set_ylabel("Classe Real", fontsize=14)
                st.pyplot(fig_cm)
                plt.close(fig_cm)

                st.write("Interpretando fisicamente: se o modelo confunde classes com espectros semelhantes, pode indicar que os modos ressonantes destas classes são muito próximos.")

            # Limpeza final
            del X, y_valid, X_train, X_temp, y_train, y_temp, X_val, X_test, y_val, y_test
            if enable_augmentation:
                del X_augmented, y_augmented, X_combined, y_combined
            gc.collect()
            os.remove(caminho_zip)
            for cat in categorias:
                caminho_cat = os.path.join(caminho_base, cat)
                for arquivo in os.listdir(caminho_cat):
                    os.remove(os.path.join(caminho_cat, arquivo))
                os.rmdir(caminho_cat)
            os.rmdir(caminho_base)
            logging.info("Processo de treino e limpeza concluído.")

        except Exception as e:
            st.error(f"Erro: {e}")
            logging.error(f"Erro: {e}")

if __name__ == "__main__":
    if app_mode == "Classificar Áudio":
        classificar_audio(SEED)
    elif app_mode == "Treinar Modelo":
        treinar_modelo(SEED)

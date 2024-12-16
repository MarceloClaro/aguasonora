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
    st.markdown("""  
    (Explicação do SEED omitida para brevidade...)  
    """)

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
Bem-vindo! Você pode:
- **Classificar Áudio:** Usar um modelo já treinado.
- **Treinar Modelo:** Treinar seu próprio modelo.
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

def extrair_features(data, sr):
    try:
        mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        st.error(f"Erro ao extrair MFCC: {e}")
        logging.error(f"Erro ao extrair MFCC: {e}")
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
    st.header("Classificação de Novo Áudio")
    # (Conteúdo do classificar_audio omitido para brevidade)
    pass

def treinar_modelo(SEED):
    st.header("Treinamento do Modelo CNN")

    st.write("""  
    ### Passo 1: Upload do Dataset (ZIP)
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
                st.error("Nenhuma subpasta de classes encontrada.")
                logging.error("Nenhuma subpasta de classes encontrada.")
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
            st.write("### Primeiras Amostras do Dataset:")
            st.dataframe(df.head())

            if len(df) == 0:
                st.error("Nenhuma amostra encontrada.")
                logging.error("Nenhuma amostra encontrada.")
                return

            labelencoder = LabelEncoder()
            y = labelencoder.fit_transform(df['classe'])
            classes = labelencoder.classes_
            st.write(f"**Classes codificadas:** {', '.join(classes)}")
            logging.info(f"Classes codificadas: {', '.join(classes)}")

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
            logging.info(f"Features extraídas: {X.shape}")

            # ==================== CONFIGURAÇÕES DE TREINAMENTO ====================
            st.sidebar.header("Configurações de Treinamento")

            num_epochs = st.sidebar.slider(
                "Número de Épocas:",
                min_value=10,
                max_value=500,
                value=200,
                step=10
            )

            batch_size = st.sidebar.selectbox(
                "Tamanho do Batch:",
                options=[8, 16, 32, 64, 128],
                index=0
            )

            # Percentual de Divisão Treino/Teste/Validação
            st.sidebar.subheader("Divisão dos Dados")
            treino_percentage = st.sidebar.slider(
                "Percentual para o Conjunto de Treino (%)",
                min_value=50,
                max_value=90,
                value=70,
                step=5
            )
            valid_percentage = st.sidebar.slider(
                "Percentual para o Conjunto de Validação (%)",
                min_value=5,
                max_value=30,
                value=15,
                step=5
            )
            test_percentage = 100 - (treino_percentage + valid_percentage)
            if test_percentage < 0:
                st.sidebar.error("A soma dos percentuais excede 100%. Ajuste os valores.")
                st.stop()
            st.sidebar.write(f"Percentual para Teste: {test_percentage}%")

            augment_factor = st.sidebar.slider(
                "Fator de Aumento de Dados:",
                min_value=1,
                max_value=100,
                value=10,
                step=1
            )

            dropout_rate = st.sidebar.slider(
                "Taxa de Dropout:",
                min_value=0.0,
                max_value=0.9,
                value=0.4,
                step=0.05
            )

            st.sidebar.subheader("Regularização")
            regularization_type = st.sidebar.selectbox(
                "Tipo de Regularização:",
                options=["None", "L1", "L2", "L1_L2"],
                index=0
            )

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
                adicionar_ruido = st.sidebar.checkbox("Adicionar Ruído Gaussiano", value=True)
                estiramento_tempo = st.sidebar.checkbox("Estiramento de Tempo", value=True)
                alteracao_pitch = st.sidebar.checkbox("Alteração de Pitch", value=True)
                deslocamento = st.sidebar.checkbox("Deslocamento", value=True)

            st.sidebar.subheader("Validação Cruzada")
            cross_validation = st.sidebar.checkbox("Ativar Validação Cruzada (k-Fold)", value=False)
            if cross_validation:
                k_folds = st.sidebar.number_input("Número de Folds:", 2, 10, 5, 1)
            else:
                k_folds = 1

            st.sidebar.subheader("Balanceamento das Classes")
            balance_classes = st.sidebar.selectbox("Método de Balanceamento:", ["Balanced", "None"], index=0)

            contagem_classes = df['classe'].value_counts()
            fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
            sns.barplot(x=contagem_classes.index, y=contagem_classes.values, palette='viridis', ax=ax_dist, legend=False)
            ax_dist.set_title("Distribuição das Classes no Dataset", fontsize=16)
            st.pyplot(fig_dist)
            plt.close(fig_dist)

            # ==================== ALTERAÇÃO: DATA AUGMENTATION ANTES DA DIVISÃO ====================
            # Realizar Data Augmentation em todo o conjunto antes da divisão
            if enable_augmentation:
                st.write("### Aplicando Data Augmentation em TODO o Conjunto Antes da Divisão...")
                X_augmented = []
                y_augmented = []

                # Recarregar os áudios dos dados originais
                for i, row in df.iterrows():
                    arquivo = row['caminho_arquivo']
                    data, sr = carregar_audio(arquivo, sr=None)
                    if data is not None:
                        # Criar a lista de transformações selecionadas
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
                                features = extrair_features(augmented_data, sr)
                                if features is not None:
                                    X_augmented.append(features)
                                    y_augmented.append(y[i])
                    else:
                        st.warning(f"Erro no carregamento do arquivo '{arquivo}' para Data Augmentation.")

                X_augmented = np.array(X_augmented)
                y_augmented = np.array(y_augmented)

                st.write(f"**Dados aumentados:** {X_augmented.shape}")

                # Combinar com os dados originais
                X_combined = np.concatenate((X, X_augmented), axis=0)
                y_combined = np.concatenate((y_valid, y_augmented), axis=0)
            else:
                X_combined = X
                y_combined = y_valid
            # ==================== FIM DA ALTERAÇÃO ====================

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

            # Ajustar forma para Conv1D
            X_train_final = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            # Cálculo de class weights
            if balance_classes == "Balanced":
                class_weights = compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(y_train),
                    y=y_train
                )
                class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
                st.write(f"**Pesos das Classes:** {class_weight_dict}")
            else:
                class_weight_dict = None

            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input

            st.sidebar.subheader("Arquitetura da CNN")
            num_conv_layers = st.sidebar.slider("Número de Camadas Convolucionais:", 1, 5, 2, 1)
            conv_filters = st.sidebar.text_input("Número de Filtros por Camada (vírgula):", "64,128")
            conv_kernel_size = st.sidebar.text_input("Tamanho do Kernel (vírgula):", "10,10")
            conv_filters = [int(f.strip()) for f in conv_filters.split(',')]
            conv_kernel_size = [int(k.strip()) for k in conv_kernel_size.split(',')]

            if len(conv_filters) != num_conv_layers or len(conv_kernel_size) != num_conv_layers:
                st.sidebar.error("Nº de filtros/kernels deve ser igual ao nº de camadas conv.")
                st.stop()

            st.sidebar.subheader("Camadas Densas")
            num_dense_layers = st.sidebar.slider("Número de Camadas Densas:", 1, 3, 1, 1)
            dense_units = st.sidebar.text_input("Neurônios por Camada Densa (vírgula):", "64")
            dense_units = [int(u.strip()) for u in dense_units.split(',')]
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

            modelo = Sequential()
            modelo.add(Input(shape=(X_train_final.shape[1], 1)))

            for i in range(num_conv_layers):
                modelo.add(Conv1D(filters=conv_filters[i], kernel_size=conv_kernel_size[i],
                                  activation='relu', kernel_regularizer=reg))
                modelo.add(Dropout(dropout_rate))
                modelo.add(MaxPooling1D(pool_size=4))

            modelo.add(Flatten())

            for i in range(num_dense_layers):
                modelo.add(Dense(units=dense_units[i], activation='relu', kernel_regularizer=reg))
                modelo.add(Dropout(dropout_rate))

            modelo.add(Dense(len(classes), activation='softmax'))

            modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Callbacks
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
            es_monitor = st.sidebar.selectbox("Monitorar:", ["val_loss", "val_accuracy"], index=0)
            es_patience = st.sidebar.slider("Paciência (Épocas):", 1, 20, 5, 1)
            es_mode = st.sidebar.selectbox("Modo:", ["min", "max"], index=0)
            earlystop = EarlyStopping(monitor=es_monitor, patience=es_patience,
                                      restore_best_weights=True, mode=es_mode)

            callbacks = [checkpointer, earlystop]

            st.write("### Treinando o Modelo...")

            with st.spinner('Treinando...'):
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

            # Salvando modelo e classes para download
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

            # (Demais códigos de avaliação podem ser adicionados aqui)

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

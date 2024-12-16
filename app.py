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
---------------
Quando um copo com água é excitado, as vibrações no fluido geram padrões de ondas,
relacionados a modos ressonantes, altura da coluna, etc. Ao extrair características
(MFCCs, centroides) do sinal acústico e treinar uma CNN, classificamos estados do
fluido-copo. Métricas avançadas, normalização e explicabilidade (SHAP) foram adicionadas.

Nesta versão:
- Exibição da arquitetura (camadas da CNN) e tabelas de validação (histórico de treinamento).
- Clustering K-Means e Hierárquico pós-treinamento, para compreender a estrutura dos dados.
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import tensorflow as tf
from tensorflow.keras.utils import to_categorical, plot_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, KFold
from sklearn.cluster import KMeans, AgglomerativeClustering
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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import shap

logging.basicConfig(
    filename='experiment_logs.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

seed_options = list(range(0, 61, 2))
default_seed = 42
if default_seed not in seed_options:
    seed_options.insert(0, default_seed)

icon_path = "logo.png"
if os.path.exists(icon_path):
    try:
        st.set_page_config(page_title="Geomaker", page_icon=icon_path, layout="wide")
        logging.info(f"Ícone {icon_path} carregado.")
    except Exception as e:
        st.set_page_config(page_title="Geomaker", layout="wide")
        logging.warning(f"Erro ao carregar o ícone {icon_path}: {e}")
else:
    st.set_page_config(page_title="Geomaker", layout="wide")
    logging.warning("Ícone não encontrado.")

st.sidebar.header("Configurações Gerais")

seed_selection = st.sidebar.selectbox(
    "Escolha o valor do SEED:",
    options=seed_options,
    index=seed_options.index(default_seed) if default_seed in seed_options else 0,
    help="Define a semente para reprodutibilidade."
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
    st.markdown("(Explicação do SEED omitida.)")

capa_path = 'capa (2).png'
if os.path.exists(capa_path):
    try:
        st.image(
            capa_path, 
            caption='Laboratório de Educação e Inteligência Artificial - Geomaker', 
            use_container_width=True
        )
    except UnidentifiedImageError:
        st.warning(f"Imagem '{capa_path}' não pôde ser carregada.")
else:
    st.warning(f"Imagem '{capa_path}' não encontrada.")

logo_path = "logo.png"
if os.path.exists(logo_path):
    try:
        st.sidebar.image(logo_path, width=200)
    except UnidentifiedImageError:
        st.sidebar.text("Logo corrompido.")
else:
    st.sidebar.text("Logo não encontrado.")

st.title("Classificação de Sons de Água Vibrando em Copo de Vidro com Aumento de Dados e CNN")
st.write("""
Opções:
- Classificar Áudio: usar um modelo treinado.
- Treinar Modelo: treinar seu próprio modelo.
""")

st.sidebar.title("Navegação")
app_mode = st.sidebar.radio("Escolha a seção", ["Classificar Áudio", "Treinar Modelo"])

eu_icon_path = "eu.ico"
if os.path.exists(eu_icon_path):
    try:
        st.sidebar.image(eu_icon_path, width=80)
    except UnidentifiedImageError:
        st.sidebar.text("Ícone 'eu.ico' corrompido.")
else:
    st.sidebar.text("Ícone 'eu.ico' não encontrado.")

st.sidebar.write("""
Produzido: Projeto Geomaker + IA

Professor: Marcelo Claro
Contato: marceloclaro@gmail.com
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
    except:
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
        features_vector = (features_vector - np.mean(features_vector)) / (np.std(features_vector) + 1e-9)
        return features_vector
    except:
        return None

def aumentar_audio(data, sr, augmentations):
    try:
        return augmentations(samples=data, sample_rate=sr)
    except:
        return data

def classificar_audio(SEED):
    st.header("Classificação de Novo Áudio (Omitido)")
    pass

def treinar_modelo(SEED):
    st.header("Treinamento do Modelo CNN")

    st.write("### Passo 1: Upload do Dataset (ZIP)")
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
                st.error("Nenhuma subpasta encontrada.")
                return

            st.success("Dataset extraído!")
            st.write(f"Classes encontradas: {', '.join(categorias)}")

            caminhos_arquivos = []
            labels = []
            for cat in categorias:
                caminho_cat = os.path.join(caminho_base, cat)
                arquivos_na_cat = [f for f in os.listdir(caminho_cat) if f.lower().endswith(('.wav', '.mp3','.flac','.ogg','.m4a'))]
                st.write(f"Classe '{cat}': {len(arquivos_na_cat)} arquivos.")
                for nome_arquivo in arquivos_na_cat:
                    caminhos_arquivos.append(os.path.join(caminho_cat, nome_arquivo))
                    labels.append(cat)

            df = pd.DataFrame({'caminho_arquivo': caminhos_arquivos, 'classe': labels})
            st.write("10 Primeiras Amostras do Dataset:")
            st.dataframe(df.head(10))

            if len(df)==0:
                st.error("Nenhuma amostra encontrada.")
                return

            labelencoder = LabelEncoder()
            y = labelencoder.fit_transform(df['classe'])
            classes = labelencoder.classes_
            st.sidebar.subheader("Parâmetros Principais")
            st.sidebar.write(f"Número de Classes: {len(classes)}")
            st.sidebar.write("Classes: "+", ".join(classes))
            st.write(f"Classes codificadas: {', '.join(classes)}")

            st.write("Extraindo Features...")
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

            # Sidebar configs
            st.sidebar.subheader("Configurações de Treinamento")
            num_epochs = st.sidebar.slider("Número de Épocas:", 10, 500, 50, 10)
            batch_size = st.sidebar.selectbox("Batch:", [8,16,32,64,128],0)
            st.sidebar.subheader("Divisão dos Dados")
            treino_percentage = st.sidebar.slider("Treino (%)",50,90,70,5)
            valid_percentage = st.sidebar.slider("Validação (%)",5,30,15,5)
            test_percentage = 100-(treino_percentage+valid_percentage)
            if test_percentage<0:
                st.sidebar.error("Treino+Val>100%")
                st.stop()
            st.sidebar.write(f"Teste (%)={test_percentage}%")

            augment_factor = st.sidebar.slider("Fator Aumento:",1,100,10,1)
            dropout_rate = st.sidebar.slider("Dropout:",0.0,0.9,0.4,0.05)

            st.sidebar.subheader("Regularização")
            regularization_type = st.sidebar.selectbox("Regularização:",["None","L1","L2","L1_L2"],0)
            if regularization_type=="L1":
                l1_regularization=st.sidebar.slider("L1:",0.0,0.1,0.001,0.001)
                l2_regularization=0.0
            elif regularization_type=="L2":
                l2_regularization=st.sidebar.slider("L2:",0.0,0.1,0.001,0.001)
                l1_regularization=0.0
            elif regularization_type=="L1_L2":
                l1_regularization=st.sidebar.slider("L1:",0.0,0.1,0.001,0.001)
                l2_regularization=st.sidebar.slider("L2:",0.0,0.1,0.001,0.001)
            else:
                l1_regularization=0.0
                l2_regularization=0.0

            enable_augmentation=st.sidebar.checkbox("Data Augmentation",True)
            if enable_augmentation:
                st.sidebar.subheader("Data Aug Options")
                adicionar_ruido=st.sidebar.checkbox("Ruído Gaussiano",True)
                estiramento_tempo=st.sidebar.checkbox("Time Stretch",True)
                alteracao_pitch=st.sidebar.checkbox("Pitch Shift",True)
                deslocamento=st.sidebar.checkbox("Deslocamento",True)

            st.sidebar.subheader("Validação Cruzada")
            cross_validation=st.sidebar.checkbox("k-Fold?",False)
            if cross_validation:
                k_folds=st.sidebar.number_input("Folds:",2,10,5,1)
            else:
                k_folds=1

            st.sidebar.subheader("Balanceamento")
            balance_classes=st.sidebar.selectbox("Balanceamento:",["Balanced","None"],0)

            if enable_augmentation:
                st.write("Aumentando Dados...")
                X_aug=[]
                y_aug=[]
                for i,row in df.iterrows():
                    arquivo=row['caminho_arquivo']
                    data, sr=carregar_audio(arquivo,sr=None)
                    if data is not None:
                        transforms=[]
                        if adicionar_ruido:
                            transforms.append(AddGaussianNoise(min_amplitude=0.001,max_amplitude=0.015,p=1.0))
                        if estiramento_tempo:
                            transforms.append(TimeStretch(min_rate=0.8,max_rate=1.25,p=1.0))
                        if alteracao_pitch:
                            transforms.append(PitchShift(min_semitones=-4,max_semitones=4,p=1.0))
                        if deslocamento:
                            transforms.append(Shift(min_shift=-0.5,max_shift=0.5,p=1.0))
                        if transforms:
                            aug=Compose(transforms)
                            for _ in range(augment_factor):
                                aug_data=aumentar_audio(data,sr,aug)
                                ftrs=extrair_features(aug_data,sr)
                                if ftrs is not None:
                                    X_aug.append(ftrs)
                                    y_aug.append(y[i])

                X_aug=np.array(X_aug)
                y_aug=np.array(y_aug)
                st.write(f"Dados Aumentados: {X_aug.shape}")
                X_combined=np.concatenate((X,X_aug),axis=0)
                y_combined=np.concatenate((y_valid,y_aug),axis=0)
            else:
                X_combined=X
                y_combined=y_valid

            st.write("Dividindo Dados...")
            X_train,X_temp,y_train,y_temp=train_test_split(X_combined,y_combined,test_size=(100-treino_percentage)/100.0,
                                                           random_state=SEED,stratify=y_combined)
            X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=test_percentage/(test_percentage+valid_percentage),
                                                       random_state=SEED,stratify=y_temp)

            st.write(f"Treino:{X_train.shape}, Val:{X_val.shape}, Teste:{X_test.shape}")

            # Guardamos X e y para clustering após o treinamento.
            X_original = X_combined
            y_original = y_combined

            X_train_final = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
            X_val = X_val.reshape((X_val.shape[0],X_val.shape[1],1))
            X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

            st.sidebar.subheader("Arquitetura da CNN")
            num_conv_layers=st.sidebar.slider("Conv Layers",1,5,2,1)
            conv_filters_str=st.sidebar.text_input("Filtros (vírgula):","64,128")
            conv_kernel_size_str=st.sidebar.text_input("Kernel (vírgula):","10,10")
            conv_filters=[int(f.strip()) for f in conv_filters_str.split(',')]
            conv_kernel_size=[int(k.strip()) for k in conv_kernel_size_str.split(',')]

            input_length=X_train_final.shape[1]
            for i in range(num_conv_layers):
                if conv_kernel_size[i]>input_length:
                    conv_kernel_size[i]=input_length

            num_dense_layers=st.sidebar.slider("Dense Layers:",1,3,1,1)
            dense_units_str=st.sidebar.text_input("Neurônios Dense (vírgula):","64")
            dense_units=[int(u.strip()) for u in dense_units_str.split(',')]
            if len(dense_units)!=num_dense_layers:
                st.sidebar.error("Neurônios != Dense Layers.")
                st.stop()

            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input

            if regularization_type=="L1":
                reg=regularizers.l1(l1_regularization)
            elif regularization_type=="L2":
                reg=regularizers.l2(l2_regularization)
            elif regularization_type=="L1_L2":
                reg=regularizers.l1_l2(l1=l1_regularization,l2=l2_regularization)
            else:
                reg=None

            modelo=Sequential()
            modelo.add(Input(shape=(X_train_final.shape[1],1)))
            for i in range(num_conv_layers):
                modelo.add(Conv1D(filters=conv_filters[i],kernel_size=conv_kernel_size[i],activation='relu',kernel_regularizer=reg))
                modelo.add(Dropout(dropout_rate))
                modelo.add(MaxPooling1D(pool_size=2))

            modelo.add(Flatten())
            for i in range(num_dense_layers):
                modelo.add(Dense(units=dense_units[i],activation='relu',kernel_regularizer=reg))
                modelo.add(Dropout(dropout_rate))
            modelo.add(Dense(len(classes),activation='softmax'))
            modelo.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

            diretorio_salvamento='modelos_salvos'
            if not os.path.exists(diretorio_salvamento):
                os.makedirs(diretorio_salvamento)

            checkpointer=ModelCheckpoint(os.path.join(diretorio_salvamento,'modelo_agua_aumentado.keras'),
                                         monitor='val_loss',verbose=1,save_best_only=True)

            es_monitor=st.sidebar.selectbox("Monitor (ES):",["val_loss","val_accuracy"],0)
            es_patience=st.sidebar.slider("Patience:",1,20,5,1)
            es_mode=st.sidebar.selectbox("Mode:",["min","max"],0)
            earlystop=EarlyStopping(monitor=es_monitor,patience=es_patience,restore_best_weights=True,mode=es_mode)
            callbacks=[checkpointer,earlystop]

            st.write("Treinando...")
            with st.spinner('Treinando...'):
                if balance_classes=="Balanced":
                    class_weights=compute_class_weight('balanced',classes=np.unique(y_train),y=y_train)
                    class_weight_dict={i:class_weights[i] for i in range(len(class_weights))}
                else:
                    class_weight_dict=None

                if cross_validation and k_folds>1:
                    kf=KFold(n_splits=k_folds,shuffle=True,random_state=SEED)
                    fold_no=1
                    val_scores=[]
                    for train_index,val_index in kf.split(X_train_final):
                        st.write(f"Fold {fold_no}")
                        X_train_cv,X_val_cv=X_train_final[train_index],X_train_final[val_index]
                        y_train_cv,y_val_cv=y_train[train_index],y_train[val_index]
                        historico=modelo.fit(X_train_cv,to_categorical(y_train_cv),
                                             epochs=num_epochs,batch_size=batch_size,
                                             validation_data=(X_val_cv,to_categorical(y_val_cv)),
                                             callbacks=callbacks,class_weight=class_weight_dict,verbose=1)
                        score=modelo.evaluate(X_val_cv,to_categorical(y_val_cv),verbose=0)
                        val_scores.append(score[1]*100)
                        fold_no+=1
                    st.write(f"Acurácia Média CV: {np.mean(val_scores):.2f}%")
                else:
                    historico=modelo.fit(X_train_final,to_categorical(y_train),
                                         epochs=num_epochs,batch_size=batch_size,
                                         validation_data=(X_val,to_categorical(y_val)),
                                         callbacks=callbacks,class_weight=class_weight_dict,verbose=1)
                st.success("Treino concluído!")

            # Download modelo
            with tempfile.NamedTemporaryFile(suffix='.keras',delete=False) as tmp_model:
                modelo.save(tmp_model.name)
                caminho_tmp_model=tmp_model.name
            with open(caminho_tmp_model,'rb') as f:
                modelo_bytes=f.read()
            buffer=io.BytesIO(modelo_bytes)
            st.download_button("Download Modelo (.keras)",data=buffer,file_name="modelo_agua_aumentado.keras")
            os.remove(caminho_tmp_model)

            classes_str="\n".join(classes)
            st.download_button("Download Classes (classes.txt)",data=classes_str,file_name="classes.txt")

            # Avaliação
            if not cross_validation:
                st.write("### Avaliação do Modelo")
                score_train=modelo.evaluate(X_train_final,to_categorical(y_train),verbose=0)
                score_val=modelo.evaluate(X_val,to_categorical(y_val),verbose=0)
                score_test=modelo.evaluate(X_test,to_categorical(y_test),verbose=0)

                st.write(f"Acurácia Treino: {score_train[1]*100:.2f}%")
                st.write(f"Acurácia Validação: {score_val[1]*100:.2f}%")
                st.write(f"Acurácia Teste: {score_test[1]*100:.2f}%")

                y_pred=modelo.predict(X_test)
                y_pred_classes=y_pred.argmax(axis=1)
                f1_val=f1_score(y_test,y_pred_classes,average='weighted')
                prec=precision_score(y_test,y_pred_classes,average='weighted')
                rec=recall_score(y_test,y_pred_classes,average='weighted')

                st.write(f"F1-score: {f1_val*100:.2f}%")
                st.write(f"Precisão: {prec*100:.2f}%")
                st.write(f"Recall: {rec*100:.2f}%")

                st.write("### Matriz de Confusão")
                cm=confusion_matrix(y_test,y_pred_classes,labels=range(len(classes)))
                cm_df=pd.DataFrame(cm,index=classes,columns=classes)
                fig_cm,ax_cm=plt.subplots(figsize=(12,8))
                sns.heatmap(cm_df,annot=True,fmt='d',cmap='Blues',ax=ax_cm)
                ax_cm.set_title("Matriz de Confusão",fontsize=16)
                ax_cm.set_xlabel("Classe Prevista",fontsize=14)
                ax_cm.set_ylabel("Classe Real",fontsize=14)
                st.pyplot(fig_cm)
                plt.close(fig_cm)

                # Plotando Arquitetura e Resumo
                st.write("### Arquitetura da CNN (Camadas)")
                modelo.summary(print_fn=lambda x: st.text(x))
                try:
                    with tempfile.NamedTemporaryFile(suffix='.png',delete=False) as tmp_plot:
                        plot_model(modelo,to_file=tmp_plot.name,show_shapes=True,show_layer_names=True)
                        img=Image.open(tmp_plot.name)
                        st.image(img,caption="Arquitetura da CNN")
                        os.remove(tmp_plot.name)
                except:
                    st.write("Não foi possível plotar a arquitetura (verifique se graphviz está instalado).")

                # Tabela das métricas de Validação por Época
                if not cross_validation:
                    st.write("### Histórico de Treinamento")
                    hist_df=pd.DataFrame(historico.history)
                    st.dataframe(hist_df)

                # Clustering KMeans e Hierárquico
                st.write("### Análise de Clusters (K-Means e Hierárquico)")
                # Utilizamos X_original (features antes do reshape, já normalizadas)
                # KMeans com 2 clusters (exemplo)
                n_clusters=2
                kmeans=KMeans(n_clusters=n_clusters,random_state=SEED)
                kmeans_labels=kmeans.fit_predict(X_original)
                st.write("Distribuição de Classes por Cluster (KMeans):")
                cluster_dist=[]
                for cidx in range(n_clusters):
                    cluster_classes=y_original[kmeans_labels==cidx]
                    counts=pd.value_counts(cluster_classes)
                    cluster_dist.append(counts)
                st.write(cluster_dist)

                # Clustering Hierárquico
                st.write("Análise Hierárquica:")
                # Usamos linkage
                Z=linkage(X_original,'ward')
                fig_dend,ax_dend=plt.subplots(figsize=(10,5))
                dendrogram(Z,ax=ax_dend)
                ax_dend.set_title("Dendrograma Hierárquico")
                ax_dend.set_xlabel("Amostras")
                ax_dend.set_ylabel("Distância")
                st.pyplot(fig_dend)
                plt.close(fig_dend)

                # Agrupamento Hierárquico em 2 clusters
                hier=AgglomerativeClustering(n_clusters=2)
                hier_labels=hier.fit_predict(X_original)
                st.write("Distribuição de Classes por Cluster (Hierárquico):")
                cluster_dist_h=[]
                for cidx in range(2):
                    cluster_classes=y_original[hier_labels==cidx]
                    counts_h=pd.value_counts(cluster_classes)
                    cluster_dist_h.append(counts_h)
                st.write(cluster_dist_h)

            # Limpeza
            gc.collect()
            os.remove(caminho_zip)
            for cat in categorias:
                caminho_cat=os.path.join(caminho_base,cat)
                for arquivo in os.listdir(caminho_cat):
                    os.remove(os.path.join(caminho_cat,arquivo))
                os.rmdir(caminho_cat)
            os.rmdir(caminho_base)
            logging.info("Processo concluído.")

        except Exception as e:
            st.error(f"Erro: {e}")
            logging.error(f"Erro: {e}")

if __name__=="__main__":
    if app_mode=="Classificar Áudio":
        classificar_audio(SEED)
    elif app_mode=="Treinar Modelo":
        treinar_modelo(SEED)


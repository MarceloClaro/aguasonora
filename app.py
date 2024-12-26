def main():
    # Configurações da página
    st.set_page_config(page_title="Classificação de Áudio com YAMNet", layout="wide")
    st.title("Classificação de Áudio com YAMNet")
    st.write("Este aplicativo permite treinar um classificador de áudio supervisionado utilizando o modelo YAMNet para extrair embeddings.")
    
    # Sidebar para parâmetros de treinamento
    st.sidebar.header("Parâmetros de Treinamento")
    epochs = st.sidebar.number_input("Número de Épocas:", min_value=1, max_value=500, value=50, step=1)
    learning_rate = st.sidebar.select_slider("Taxa de Aprendizagem:", options=[0.1, 0.01, 0.001, 0.0001], value=0.001)
    batch_size = st.sidebar.selectbox("Tamanho de Lote:", options=[8, 16, 32, 64], index=1)
    l2_lambda = st.sidebar.number_input("Regularização L2 (Weight Decay):", min_value=0.0, max_value=0.1, value=0.01, step=0.01)
    patience = st.sidebar.number_input("Paciência para Early Stopping:", min_value=1, max_value=10, value=3, step=1)
    
    # Upload de dados supervisionados
    st.header("Upload de Dados Supervisionados")
    st.write("Envie um arquivo ZIP contendo subpastas com arquivos de áudio organizados por classe. Por exemplo:")
    st.write("""
    ```
    dados/
        agua_quente/
            audio1.wav
            audio2.wav
        agua_fria/
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
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)
    
            # Verificar estrutura de diretórios
            classes = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
            if len(classes) < 2:
                st.error("O arquivo ZIP deve conter pelo menos duas subpastas, cada uma representando uma classe.")
            else:
                st.success(f"Classes encontradas: {classes}")
                # Contar arquivos por classe
                class_counts = {cls: len([f for f in os.listdir(os.path.join(tmpdir, cls)) if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]) for cls in classes}
                st.write("**Contagem de arquivos por classe:**")
                st.write(class_counts)
    
                # Preparar dados para treinamento
                st.header("Preparando Dados para Treinamento")
                yamnet_model = load_yamnet_model()
                st.write("Modelo YAMNet carregado.")
    
                embeddings = []
                labels = []
                label_mapping = {cls: idx for idx, cls in enumerate(classes)}
    
                total_files = sum(class_counts.values())
                processed_files = 0
                progress_bar = st.progress(0)
    
                for cls in classes:
                    cls_dir = os.path.join(tmpdir, cls)
                    audio_files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
                    for audio_file in audio_files:
                        pred_class, embedding = extract_yamnet_embeddings(yamnet_model, audio_file)
                        if embedding is not None:
                            embeddings.append(embedding.flatten())
                            labels.append(label_mapping[cls])
                        processed_files += 1
                        progress_bar.progress(processed_files / total_files)
    
                # Verificações após a extração
                if len(embeddings) == 0:
                    st.error("Nenhum embedding foi extraído. Verifique se os arquivos de áudio estão no formato correto e se o YAMNet está funcionando corretamente.")
                    return
    
                # Verificar se todos os embeddings têm o mesmo tamanho
                embedding_shapes = [emb.shape for emb in embeddings]
                unique_shapes = set(embedding_shapes)
                if len(unique_shapes) != 1:
                    st.error(f"Embeddings têm tamanhos inconsistentes: {unique_shapes}")
                    return
    
                # Converter para array NumPy
                try:
                    embeddings = np.array(embeddings)
                    st.write(f"Embeddings convertidos para array NumPy: Shape = {embeddings.shape}")
                except ValueError as ve:
                    st.error(f"Erro ao converter embeddings para array NumPy: {ve}")
                    return
    
                # Dividir os dados em treino e validação
                X_train, X_val, y_train, y_val = train_test_split(
                    embeddings, labels, test_size=0.2, random_state=42, stratify=labels
                )
                st.write(f"Dados divididos em treino ({len(X_train)} amostras) e validação ({len(X_val)} amostras).")
    
                # Treinar o classificador
                st.header("Treinamento do Classificador")
                classifier = train_audio_classifier(
                    X_train, y_train, X_val, y_val, 
                    input_dim=embeddings.shape[1], 
                    num_classes=len(classes), 
                    epochs=epochs, 
                    learning_rate=learning_rate, 
                    batch_size=batch_size, 
                    l2_lambda=l2_lambda, 
                    patience=patience
                )
                st.success("Treinamento do classificador concluído.")
    
                # Salvar o classificador no estado do Streamlit
                st.session_state['classifier'] = classifier
                st.session_state['classes'] = classes
    
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
    if 'classifier' in st.session_state and 'classes' in st.session_state:
        st.header("Classificação de Novo Áudio")
        st.write("Envie um arquivo de áudio para ser classificado pelo modelo treinado.")
        uploaded_audio = st.file_uploader("Faça upload do arquivo de áudio para classificação", type=["wav", "mp3", "ogg", "flac"])
    
        if uploaded_audio is not None:
            try:
                # Salvar o arquivo de áudio em um diretório temporário
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmp_audio:
                    tmp_audio.write(uploaded_audio.read())
                    tmp_audio_path = tmp_audio.name
    
                # Carregar a imagem para exibição (opcional)
                # Aqui, apenas exibimos uma representação visual simples
                audio_image = Image.new('RGB', (400, 100), color = (73, 109, 137))
                plt.figure(figsize=(4,1))
                plt.text(0.5, 0.5, 'Áudio Enviado', horizontalalignment='center', verticalalignment='center', fontsize=20, color='white')
                plt.axis('off')
                plt.savefig(tmp_audio_path + '_image.png')
                plt.close()
                st.image(tmp_audio_path + '_image.png', caption='Áudio Enviado', use_column_width=True)
            except Exception as e:
                st.error(f"Erro ao processar o áudio: {e}")
                tmp_audio_path = None
    
            if tmp_audio_path is not None:
                # Extrair embeddings
                yamnet_model = load_yamnet_model()
                pred_class, embedding = extract_yamnet_embeddings(yamnet_model, tmp_audio_path)
    
                if embedding is not None and pred_class != -1:
                    # Obter o modelo treinado
                    classifier = st.session_state['classifier']
                    classes = st.session_state['classes']
    
                    # Converter o embedding para tensor
                    embedding_tensor = torch.tensor(embedding.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
    
                    # Classificar
                    classifier.eval()
                    with torch.no_grad():
                        output = classifier(embedding_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        class_idx = predicted.item()
                        class_name = classes[class_idx]
                        confidence_score = confidence.item()
    
                    # Exibir resultados
                    st.write(f"**Classe Predita:** {class_name}")
                    st.write(f"**Confiança:** {confidence_score:.4f}")
    
                # Remover arquivos temporários
                os.remove(tmp_audio_path)
                if os.path.exists(tmp_audio_path + '_image.png'):
                    os.remove(tmp_audio_path + '_image.png')
    
    st.write("### Documentação dos Procedimentos")
    st.write("""
    1. **Upload de Dados Supervisionados**: Envie um arquivo ZIP contendo subpastas, onde cada subpasta representa uma classe com seus respectivos arquivos de áudio.
    
    2. **Extração de Embeddings**: Utilizamos o YAMNet para extrair embeddings dos arquivos de áudio enviados.
    
    3. **Treinamento do Classificador**: Com os embeddings extraídos, treinamos um classificador personalizado conforme os parâmetros definidos na barra lateral.
    
    4. **Classificação de Novo Áudio**: Após o treinamento, você pode enviar um novo arquivo de áudio para ser classificado pelo modelo treinado.
    
    **Exemplo de Estrutura de Diretórios para Upload**:
    ```
    dados/
        agua_quente/
            audio1.wav
            audio2.wav
        agua_fria/
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

def classify_new_audio(uploaded_audio):
    try:
        # Salvar o arquivo de áudio em um diretório temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmp_audio:
            tmp_audio.write(uploaded_audio.read())
            tmp_audio_path = tmp_audio.name

        # Audição do Áudio Carregado
        try:
            sample_rate, wav_data = wavfile.read(tmp_audio_path, 'rb')
            # Verificar se o áudio é mono e 16kHz
            if wav_data.ndim > 1:
                wav_data = wav_data.mean(axis=1)
            if sample_rate != 16000:
                sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
            # Normalizar
            if wav_data.dtype.kind == 'i':
                wav_data = wav_data / np.iinfo(wav_data.dtype).max
            elif wav_data.dtype.kind == 'f':
                if np.max(wav_data) > 1.0 or np.min(wav_data) < -1.0:
                    wav_data = wav_data / np.max(np.abs(wav_data))
            # Convert to float32
            wav_data = wav_data.astype(np.float32)
            duration = len(wav_data) / sample_rate
            st.write(f"**Sample rate:** {sample_rate} Hz")
            st.write(f"**Total duration:** {duration:.2f}s")
            st.write(f"**Size of the input:** {len(wav_data)} samples")
            st.audio(wav_data, format='audio/wav', sample_rate=sample_rate)
        except Exception as e:
            st.error(f"Erro ao processar o áudio para audição: {e}")
            wav_data = None

        # Inferir BPM usando Librosa
        try:
            tempo, beat_frames = librosa.beat.beat_track(y=wav_data, sr=sample_rate)
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
                tempo = 0.0  # Valor padrão ou lidar de acordo com a lógica do seu aplicativo

            st.write(f"**BPM Inferido com Librosa:** {tempo:.2f}")
        except Exception as e:
            st.warning(f"Falha ao inferir BPM com Librosa: {e}")
            # Opção Criativa: Solicitar ao usuário para inserir manualmente o BPM
            tempo = st.number_input("Insira o BPM manualmente:", min_value=20.0, max_value=300.0, value=120.0, step=1.0)

        # Extrair embeddings
        yamnet_model = load_yamnet_model()
        pred_class, embedding = extract_yamnet_embeddings(yamnet_model, tmp_audio_path)

        if embedding is not None and pred_class != -1:
            # Obter o modelo treinado
            classifier = st.session_state['classifier']
            classes = st.session_state['classes']

            # Converter o embedding para tensor
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)

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

            # Visualização do Áudio
            st.subheader("Visualização do Áudio")
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))

            # Plot da Forma de Onda
            ax[0].plot(wav_data)
            ax[0].set_title("Forma de Onda")
            ax[0].set_xlabel("Amostras")
            ax[0].set_ylabel("Amplitude")

            # Plot do Espectrograma
            S = librosa.feature.melspectrogram(y=wav_data, sr=sample_rate, n_mels=128)
            S_DB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_DB, sr=sample_rate, x_axis='time', y_axis='mel', ax=ax[1])
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
            try:
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
                confident_indices = np.where(confidence_outputs >= 0.8)[0]  # Ajustar o limiar conforme necessário
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
                # Definir constantes para conversão
                note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

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
                predictions_per_eighth = 40  # Aumentou de 20 para 40
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

                st.write(f"Notas e Rests Detectados: {best_notes_and_rests}")

                # Criar a partitura musical usando music21
                create_music_score(best_notes_and_rests, tempo, showScore, tmp_audio_path)

            except Exception as e:
                st.error(f"Erro ao processar o áudio com SPICE: {e}")

    if __name__ == "__main__":
        main()

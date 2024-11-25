from flask import Flask, request, jsonify
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from the Flutter app

# File paths (relative paths for Docker container)
model_directory = '/app/model'  # Place the models in /app/models inside the container

# Initialize global variables for tokenizers and models to keep them in memory
tokenizers = {}
models = {}

# Function to load tokenizers and models based on the source and target languages
def load_tokenizers_and_model(source_lang, target_lang):
    # Cache the tokenizers and models to avoid reloading them multiple times
    if (source_lang, target_lang) in models:
        return tokenizers[(source_lang, target_lang)], models[(source_lang, target_lang)]

    if source_lang == 'Tagalog':
        src_tokenizer_path = os.path.join(model_directory, 'source_tokenizer_taga_cuyo.json')
        target_tokenizer_path = os.path.join(model_directory, 'target_tokenizer_taga_cuyo.json')
        model_path = os.path.join(model_directory, 'model_taga_cuyo.keras')
        encoder_max_length = 13
        decoder_max_length = 17
    else:
        src_tokenizer_path = os.path.join(model_directory, 'source_tokenizer_cuyo_taga.json')
        target_tokenizer_path = os.path.join(model_directory, 'target_tokenizer_cuyo_taga.json')
        model_path = os.path.join(model_directory, 'model_cuyo_taga.keras')
        encoder_max_length = 16
        decoder_max_length = 14

    # Load tokenizers
    with open(src_tokenizer_path) as f:
        source_tokenizer = tokenizer_from_json(json.load(f))
    with open(target_tokenizer_path) as f:
        target_tokenizer = tokenizer_from_json(json.load(f))

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Cache the loaded tokenizers and model
    tokenizers[(source_lang, target_lang)] = (source_tokenizer, target_tokenizer)
    models[(source_lang, target_lang)] = model

    return source_tokenizer, target_tokenizer, model

# Encoder and Decoder Setup
def create_encoder_decoder_model(model):
    # Encoder Setup
    encoder_inputs = tf.keras.Input(shape=(None,), dtype="float32", name="encoder_inputs")
    encoder_embedding_layer = model.get_layer('encoder_embeddings')
    encoder_embeddings = encoder_embedding_layer(encoder_inputs)
    encoder_lstm = model.get_layer('encoder_lstm')
    _, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embeddings)
    encoder_states = [encoder_state_h, encoder_state_c]
    encoder_model_no_attention = tf.keras.Model(encoder_inputs, encoder_states)

    # Decoder Setup
    decoder_inputs = tf.keras.Input(shape=(None,), dtype='int32', name='decoder_inputs')
    decoder_embedding_layer = model.get_layer('decoder_embeddings')
    decoder_embeddings = decoder_embedding_layer(decoder_inputs)
    decoder_input_state_h = tf.keras.Input(shape=(512,), name='decoder_input_state_h')
    decoder_input_state_c = tf.keras.Input(shape=(512,), name='decoder_input_state_c')
    decoder_input_states = [decoder_input_state_h, decoder_input_state_c]
    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_sequence_outputs, decoder_output_state_h, decoder_output_state_c = decoder_lstm(
        decoder_embeddings, initial_state=decoder_input_states
    )
    decoder_output_states = [decoder_output_state_h, decoder_output_state_c]
    decoder_dense = model.get_layer('decoder_dense')
    y_proba = decoder_dense(decoder_sequence_outputs)
    decoder_model_no_attention = tf.keras.Model(
        [decoder_inputs] + decoder_input_states,
        [y_proba] + decoder_output_states
    )
    
    return encoder_model_no_attention, decoder_model_no_attention

# Translation function
def translate_without_attention(sentence, source_tokenizer, encoder, target_tokenizer, decoder, max_translated_len=100):
    input_seq = source_tokenizer.texts_to_sequences([sentence])
    input_seq = tf.convert_to_tensor(input_seq, dtype=tf.float32)
    states = encoder.predict(input_seq)
    
    current_word = '<sos>'
    decoded_sentence = []

    while len(decoded_sentence) < max_translated_len:
        target_seq = np.zeros((1, 1), dtype=np.int32)
        target_seq[0, 0] = target_tokenizer.word_index[current_word]
        target_y_proba, h, c = decoder.predict([target_seq] + states)

        target_token_index = np.argmax(target_y_proba[0, -1, :])
        current_word = target_tokenizer.index_word.get(target_token_index, '<unk>')

        if current_word == '<eos>':
            break

        decoded_sentence.append(current_word)
        states = [h, c]

    return ' '.join(decoded_sentence)

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    source_lang = data.get("source_lang", "Tagalog")
    target_lang = data.get("target_lang", "Cuyonon")
    sentence = data.get("sentence", "").strip()

    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400
    
    if not sentence.endswith('.'):
        sentence += ' .'

    # Load the relevant tokenizer and model for the given language pair
    source_tokenizer, target_tokenizer, model = load_tokenizers_and_model(source_lang, target_lang)

    # Create encoder and decoder models
    encoder_model_no_attention, decoder_model_no_attention = create_encoder_decoder_model(model)

    # Call the translate function
    translated_sentence = translate_without_attention(sentence, source_tokenizer, encoder_model_no_attention, target_tokenizer, decoder_model_no_attention)
    
    return jsonify({"translation": translated_sentence})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

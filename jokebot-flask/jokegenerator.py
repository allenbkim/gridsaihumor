from flask import Flask, render_template, request, url_for
from keras.models import Model, load_model
import tensorflow as tf
import numpy as np
import pickle
import os

# Constants used in execution
APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
APP_STATIC = os.path.join(APP_ROOT, 'static')
ENCODER_MODEL_FILE = APP_ROOT + '/gridsai-qahumor-encoder-model.h5'
DECODER_MODEL_FILE = APP_ROOT + '/gridsai-qahumor-decoder-model.h5'
ENCODER_PICKLE_FILE = APP_ROOT + '/gridsai-qahumor-encoder-pickle.pckl'
DECODER_PICKLE_FILE = APP_ROOT + '/gridsai-qahumor-decoder-pickle.pckl'
CHAR_TOKENS = ['\n', '\t', ' ', '.', ',', '!', '?', ':', ';', '$', '#', '@', '%', '^', '&', '*', '(', ')', '-', '_',
               '=', '+', '\\', '/', '|', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
               'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5',
               '6', '7', '8', '9', '0', '\"', '\'', '[', ']', '{', '}', '<', '>', '`', '~', '’', 'ø', 'è', '£', 'é',
               '∫', '—', '͡', '°', '͜', 'ʖ', '']

# Global variables
encoder_model = None
decoder_model = None
graph = None

app = Flask(__name__)


def setup():
    global encoder_model
    encoder_model = load_model(ENCODER_MODEL_FILE)
    global decoder_model
    decoder_model = load_model(DECODER_MODEL_FILE)
    global graph
    graph = tf.get_default_graph()


setup()


@app.route('/', methods=['POST', 'GET'])
def generate_joke():
    if request.method == 'GET':
        return render_template('jokes.html')
    elif request.method == 'POST':
        joke = '\t' + request.form['userJoke'] + '\n'
        answer = forward_pass([joke])[0][1]
        return render_template('jokes.html', userJoke=request.form['userJoke'], punchline=answer)


def decode_sequence(input_seq, num_decoder_tokens, target_token_index, reverse_target_char_index, max_decoder_seq_length):
    with graph.as_default():
        states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        with graph.as_default():
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


def forward_pass(jokes_needing_punchlines):
    """
    Generates
    :param jokes_needing_punchlines: List of joke setup strings.
    :return: List of 2-tuples containing joke setup/generated punchline pairs.
    """
    decoded_sentences = []

    encoder_pickle_file = open(ENCODER_PICKLE_FILE, 'rb')
    max_encoder_seq_length = pickle.load(encoder_pickle_file)
    encoder_pickle_file.close()
    decoder_pickle_file = open(DECODER_PICKLE_FILE, 'rb')
    max_decoder_seq_length = pickle.load(decoder_pickle_file)
    decoder_pickle_file.close()

    # Reverse-lookup token index to decode sequences back to something readable.
    characters = sorted(list(CHAR_TOKENS))
    num_tokens = len(characters)
    token_index = dict([(char, i) for i, char in enumerate(characters)])
    reverse_char_index = dict((i, char) for char, i in token_index.items())

    encoder_input_data = np.zeros(
        (len(jokes_needing_punchlines), max_encoder_seq_length, num_tokens),
        dtype='float32')

    for i, input_text in enumerate(jokes_needing_punchlines):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, token_index[char]] = 1.

    for seq_index in range(len(jokes_needing_punchlines)):
        # Take one sequence for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq, num_tokens, token_index, reverse_char_index, max_decoder_seq_length)
        decoded_sentences.append((jokes_needing_punchlines[seq_index], decoded_sentence))

    return decoded_sentences

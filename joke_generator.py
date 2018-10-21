from __future__ import print_function

from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
import numpy as np
import pandas as pd
import pickle


# Constants used in execution
TRAIN_MODEL = False  # True to run training. False to do just forward pass.
BATCH_SIZE = 64  # Batch size for training.
EPOCHS = 5  # Number of epochs to train for.
LATENT_DIM = 256  # Latent dimensionality of the encoding space.
DATA_PATH = 'jokes.csv'  # Path to the data txt file on disk.
TRAINING_MODEL_FILE = 'gridsai-qahumor-main-model.h5'   # Name of the main model file.
ENCODER_MODEL_FILE = 'gridsai-qahumor-encoder-model.h5'
DECODER_MODEL_FILE = 'gridsai-qahumor-decoder-model.h5'
ENCODER_PICKLE_FILE = 'gridsai-qahumor-encoder-pickle.pckl'
DECODER_PICKLE_FILE = 'gridsai-qahumor-decoder-pickle.pckl'

CHAR_TOKENS = ['\n', '\t', ' ', '.', ',', '!', '?', ':', ';', '$', '#', '@', '%', '^', '&', '*', '(', ')', '-', '_',
               '=', '+', '\\', '/', '|', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
               'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5',
               '6', '7', '8', '9', '0', '\"', '\'', '[', ']', '{', '}', '<', '>', '`', '~', '’', 'ø', 'è', '£', 'é',
               '∫', '—', '͡', '°', '͜', 'ʖ', '']


def train_model():
    # Vector-ize the data.
    input_texts = []
    target_texts = []

    jokes = pd.read_csv(DATA_PATH, header=0)
    for _, row in enumerate(jokes.values):
        input_text, target_text = row[1].strip('\n'), row[2].strip('\n')
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)

    characters = sorted(list(CHAR_TOKENS))
    num_tokens = len(characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique tokens:', num_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    token_index = dict(
        [(char, i) for i, char in enumerate(characters)])

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_tokens),
        dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            if char in token_index:
                encoder_input_data[i, t, token_index[char]] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            if char in token_index:
                decoder_input_data[i, t, token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, token_index[char]] = 1.

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_tokens))
    encoder = LSTM(LATENT_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard 'encoder_outputs' and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using 'encoder_states' as initial state.
    decoder_inputs = Input(shape=(None, num_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # 'encoder_input_data' & 'decoder_input_data' into 'decoder_target_data'
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_split=0.2)
    # Save model
    model.save(TRAINING_MODEL_FILE)

    # Next: inference mode (sampling).
    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states

    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.save(ENCODER_MODEL_FILE)

    decoder_state_input_h = Input(shape=(LATENT_DIM,))
    decoder_state_input_c = Input(shape=(LATENT_DIM,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    decoder_model.save(DECODER_MODEL_FILE)

    # Pickle some data for use when decoding
    encoder_pickle_file = open(ENCODER_PICKLE_FILE, 'wb')
    pickle.dump(max_encoder_seq_length, encoder_pickle_file)
    encoder_pickle_file.close()
    decoder_pickle_file = open(DECODER_PICKLE_FILE, 'wb')
    pickle.dump(max_decoder_seq_length, decoder_pickle_file)
    decoder_pickle_file.close()


def decode_sequence(input_seq, num_decoder_tokens, target_token_index, reverse_target_char_index, max_decoder_seq_length):
    # Load up models and data
    encoder_model = load_model(ENCODER_MODEL_FILE)
    decoder_model = load_model(DECODER_MODEL_FILE)

    # Encode the input as state vectors.
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
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

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


# ###################################
# Let's get down to (funny) business!
# ###################################
# Train the model if config'd to do so
if TRAIN_MODEL:
    train_model()

# After model is trained, run forward pass with new joke setups that need punchlines
punchlines = forward_pass(['\tWhy did the chicken cross the road?\n',
                           '\tWho let the dogs out?\n',
                           '\tHow much wood would a woodchuck chuck?\n',
                           '\tWhat did the Trojan say to the Bruin?\n',
                           '\tWhat do you get when you cross a Data Scientist with an AI Engineer??\n',
                           '\tWhere did the pig go to buy some food?\n'])

# Let's see what was generated!
for punchline in punchlines:
    print('-')
    print('Input sentence:', punchline[0])
    print('Decoded sentence:', punchline[1])

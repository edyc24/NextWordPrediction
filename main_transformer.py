# data3.txt has 57,008 words and 309,690 characters

# Importing necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import (
    Embedding,
    Dense,
    Input,
    GlobalAveragePooling1D,
    MultiHeadAttention,
    LayerNormalization,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np
import os

# Importing files in Google Colab
from google.colab import files

uploaded = files.upload()

# Opening and reading data from a file
file = open("data3.txt", "r", encoding="utf8")
lines = []
for i in file:
    lines.append(i)

# Concatenating lines to form a single string
data = ""
for i in lines:
    data = " ".join(lines)

# Cleaning the data
data = (
    data.replace("\n", "")
    .replace("\r", "")
    .replace("\ufeff", "")
    .replace("“", "")
    .replace("”", "")
    .replace("_", "")
)

# Splitting data into words
data = data.split()
data = " ".join(data)
data[:500]

# Tokenizing the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# Saving the tokenizer for later use
pickle.dump(tokenizer, open("token.pk1", "wb"))

# Converting the data into sequences
sequence_data = tokenizer.texts_to_sequences([data])[0]
sequence_data[:15]

# Determining the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

# Creating sequences of three words and their next word
sequences = []
for i in range(3, len(sequence_data)):
    words = sequence_data[i - 3 : i + 1]
    sequences.append(words)

# Converting sequences to a numpy array
sequences = np.array(sequences)

# Splitting sequences into input (X) and output (Y)
X, Y = sequences[:, :-1], sequences[:, -1]

# One-hot encoding the output (Y)
Y = to_categorical(Y, num_classes=vocab_size)


# Transformer block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(
        inputs, inputs
    )
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res


# Building the model with Transformer
def build_model(vocab_size, max_len=3):
    inputs = Input(shape=(max_len,))
    embedding_layer = Embedding(vocab_size, 64)(inputs)
    x = transformer_encoder(embedding_layer, head_size=64, num_heads=2, ff_dim=4)
    x = GlobalAveragePooling1D()(x)
    x = Dense(1000, activation="relu")(x)
    outputs = Dense(vocab_size, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


model = build_model(vocab_size, max_len=3)

# Setting up ModelCheckpoint for saving the best model
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    "next_words.h5", monitor="loss", verbose=1, save_best_only=True
)

# Compiling and training the model
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001))
model.fit(X, Y, epochs=70, batch_size=64, callbacks=[checkpoint])

# Loading the trained model and tokenizer for prediction
model = load_model("next_words.h5")
tokenizer = pickle.load(open("token.pk1", "rb"))


# Function to predict the next word
def Predict_Next_Words(model, tokenizer, text, top_n=5):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)
    preds = model.predict(sequence)
    top_preds_indices = np.argsort(-preds)[0, :top_n]

    predicted_words = []
    for i in range(top_n):
        predicted_word = ""
        for key, value in tokenizer.word_index.items():
            if value == top_preds_indices[i]:
                predicted_word = key
                break
        predicted_words.append(predicted_word)

    print("Top " + str(top_n) + " possible words: " + ", ".join(predicted_words))
    return predicted_words


# Taking user input for text prediction
while True:
    text = input("Enter your line: ")
    if text == "STOP":
        print("Exiting.")
        break
    else:
        try:
            Predict_Next_Words(model, tokenizer, text)
        except Exception as e:
            print("Error:", e)
            print("Please try again.")

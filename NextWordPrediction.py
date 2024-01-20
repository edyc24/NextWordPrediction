# Importing necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
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
  data = ' '.join(lines)

# Cleaning the data
data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('“', '').replace('”', '').replace('_', '')

# Splitting data into words
data = data.split()
data = ' '.join(data)
data[:500]

# Tokenizing the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# Saving the tokenizer for later use
pickle.dump(tokenizer, open('token.pk1', 'wb'))

# Converting the data into sequences
sequence_data = tokenizer.texts_to_sequences([data])[0]
sequence_data[:15]

# Determining the vocabulary size
len(sequence_data)
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

# Creating sequences of three words and their next word
sequences = []
for i in range(3, len(sequence_data)):
  words = sequence_data[i-3:i+1]
  sequences.append(words)

# Converting sequences to a numpy array
sequences = np.array(sequences)
sequences[:10]

# Splitting sequences into input (X) and output (Y)
X = []
Y = []
for i in sequences:
  X.append(i[0:3])
  Y.append(i[3])

X = np.array(X)
Y = np.array(Y)

# One-hot encoding the output (Y)
Y = to_categorical(Y, num_classes=vocab_size)
Y[:5]

# Building the LSTM model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=3))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))

# Setting up ModelCheckpoint for saving the best model
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("next_words.h5", monitor='loss', verbose=1, save_best_only=True)

# Compiling and training the model
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001))
model.fit(X, Y, epochs=70, batch_size=64, callbacks=[checkpoint])

# Loading the trained model and tokenizer for prediction
from tensorflow.keras.models import load_model
import numpy as np
import pickle

model = load_model('next_words.h5')
tokenizer = pickle.load(open('token.pk1', 'rb'))

# Function to predict the next word
def Predict_Next_Words(model, tokenizer, text, top_n=5):
 sequence = tokenizer.texts_to_sequences([text])
 sequence = np.array(sequence)
 preds = model.predict(sequence)
 top_preds_indices = np.argsort(-preds)[0, :top_n]

 predicted_words = []
 # Mapping the predicted indices to the corresponding words
 for i in range(top_n):
    predicted_word = ""
    for key, value in tokenizer.word_index.items():
      if value == top_preds_indices[i]:
        predicted_word = key
        break
    predicted_words.append(predicted_word)

 print("Top " + str(top_n) + " possible words: " + ", ".join(predicted_words))
 return predicted_words

# Taking user input for predicting the next word
while True:
  text = input("Enter a 3 word phrase: ")

  if text == "0":
    print("Good bye!!!")
    break

  else:
    try:
      text = text.split(" ")
      text = text[-3:]
      result = Predict_Next_Words(model, tokenizer, text)
      print("Predicted next word:", result)

    except Exception as e:
      print("Error occurred: ", e)
      continue

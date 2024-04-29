import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

max_features = 10000  # Consider the top 10,000 most frequent words
maxlen = 200  # Cut reviews after 200 words

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')

model = Sequential()
model.add(Embedding(max_features, 128, input_shape = (maxlen,)))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)

loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)

new_text = ["This movie was fantastic!", "I didn't like this movie at all."]
# Assuming new_text is a list of strings containing the new movie reviews

# Tokenize and pad the new text data
x_new = pad_sequences(tokenizer.texts_to_sequences(new_text), maxlen=maxlen)

# Predict sentiment for new data
predictions = model.predict(x_new)

# Print predictions
for i, prediction in enumerate(predictions):
    print("Review:", new_text[i])
    print("Predicted Sentiment:", "Positive" if prediction >= 0.5 else "Negative")
    print("Confidence:", prediction[0])
    print()
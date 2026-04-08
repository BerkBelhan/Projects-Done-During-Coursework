import numpy as np
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.datasets import imdb

max_features = 10000
maxlen = 500
batch_size = 32
embedding_dim = 128
hidden_units = 64
epochs = 5

print("Loading data...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
print(f"{len(X_train)} train sequences")
print(f"{len(X_test)} test sequences") 

print("Pad sequences (samples x time)")
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

print("Build model...")
model = Sequential()
model.add(Embedding(max_features, embedding_dim, input_length=maxlen))
model.add(SimpleRNN(hidden_units))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Training model...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
print("Evaluating model...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

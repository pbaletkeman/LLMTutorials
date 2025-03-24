# Text classification using CNN Implementation

# importing the necessary libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Setting up the parameters
maximum_features = 5000  # Maximum number of words to consider as features
maximum_length = 100  # Maximum length of input sequences
word_embedding_dims = 50  # Dimension of word embeddings
no_of_filters = 250  # Number of filters in the convolutional layer
kernel_size = 3  # Size of the convolutional filters
hidden_dims = 250  # Number of neurons in the hidden layer
batch_size = 32  # Batch size for training
epochs = 2  # Number of training epochs
threshold = 0.5  # Threshold for binary classification

# Loading the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=maximum_features)

# Padding the sequences to ensure uniform length
x_train = pad_sequences(x_train, maxlen=maximum_length)
x_test = pad_sequences(x_test, maxlen=maximum_length)

# Building the model
model = Sequential()

# Adding the embedding layer to convert input sequences to dense vectors
model.add(Embedding(maximum_features, word_embedding_dims, input_length=maximum_length))

# Adding the 1D convolutional layer with ReLU activation
model.add(Conv1D(no_of_filters, kernel_size, padding='valid', activation='relu', strides=1))

# Adding the global max pooling layer to reduce dimensionality
model.add(GlobalMaxPooling1D())

# Adding the dense hidden layer with ReLU activation
model.add(Dense(hidden_dims, activation='relu'))

# Adding the output layer with sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compiling the model with binary cross-entropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Predicting the probabilities for test data
y_pred_prob = model.predict(x_test)

# Converting the probabilities to binary classes based on threshold
y_pred = (y_pred_prob > threshold).astype(int)

# Calculating the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Printing the evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

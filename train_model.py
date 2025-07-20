import pickle

import numpy as np
import pandas as pd
from keras.layers import Embedding, Conv1D, MaxPooling1D, GRU, Dense, Dropout, BatchNormalization
from keras.models import Sequential, load_model
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load the original dataset
original_data = pd.read_csv('cyberbullying.csv')
original_data['Text'] = original_data['Text'].fillna('')

texts_original = original_data['Text']
labels_original = (original_data['CB_Label']).astype(int)
# Tokenize the text data of the original dataset
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts_original)
vocab_size = len(tokenizer.word_index) + 1

# Load pre-trained GloVe word embeddings before tokenization
embedding_index = {}
glove_file = 'glove.6B.100d.txt'
with open(glove_file, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

# Create an embedding matrix for the original dataset
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Convert texts of the original dataset to sequences and pad them
sequences_original = tokenizer.texts_to_sequences(texts_original)
max_sequence_length = 100
padded_sequences_original = pad_sequences(sequences_original, maxlen=max_sequence_length, padding='post',
                                          truncating='post')

# Split the data into training, validation, and testing sets for the original dataset
X_train_original, X_temp_original, y_train_original, y_temp_original = train_test_split(padded_sequences_original,
                                                                                        labels_original, test_size=0.2,
                                                                                        random_state=42)
X_val_original, X_test_original, y_val_original, y_test_original = train_test_split(X_temp_original, y_temp_original,
                                                                                    test_size=0.5, random_state=42)

# Load the pre-trained model
pre_trained_model = load_model('best_model.h5')

# Load and preprocess the new dataset
new_data = pd.read_csv('New_Dataset.csv')  # Replace 'New_Dataset2.csv' with the actual filename of your new dataset

# Handle missing values in a more robust way (e.g., removing rows with missing values)
new_data = new_data.dropna(subset=['Text'])

texts_new = new_data['Text']  # 'Text' column of your new dataset
labels_new = (new_data['CB_Label']).astype(int)  # Adapt labels for binary classification

# Tokenize the text data of the new dataset using the existing tokenizer
sequences_new = tokenizer.texts_to_sequences(texts_new)
padded_sequences_new = pad_sequences(sequences_new, maxlen=max_sequence_length, padding='post', truncating='post')

# Split the data into training, validation, and testing sets for the new dataset
X_train_new, X_temp_new, y_train_new, y_temp_new = train_test_split(padded_sequences_new, labels_new, test_size=0.2,
                                                                    random_state=42)
X_val_new, X_test_new, y_val_new, y_test_new = train_test_split(X_temp_new, y_temp_new, test_size=0.5, random_state=42)

# Build a new model with the same architecture
new_model = Sequential()
new_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length,
                        weights=[embedding_matrix], trainable=False))
new_model.add(Conv1D(128, 5, activation='relu'))
new_model.add(MaxPooling1D(5))
new_model.add(GRU(64, dropout=0.5, recurrent_dropout=0.5))
new_model.add(Dense(32, activation='relu', kernel_regularizer='l2'))
new_model.add(BatchNormalization())
new_model.add(Dropout(0.7))
new_model.add(Dense(1, activation='sigmoid'))

# Compile the new model
new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the new model on the new dataset
epochs = 20
batch_size = 64
history = new_model.fit(X_train_new, y_train_new, validation_data=(X_val_new, y_val_new), epochs=epochs,
                        batch_size=batch_size)

# Evaluate the model on the test set of the new dataset
accuracy = new_model.evaluate(X_test_new, y_test_new)
print(f"Test accuracy: {accuracy[1]}")

# Print additional metrics
y_pred_new = new_model.predict(X_test_new)
y_pred_binary_new = (y_pred_new >= 0.5).astype(int)
print("Additional Metrics:")
print(classification_report(y_test_new, y_pred_binary_new))

# Save the updated model and tokenizer
new_model.save('model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

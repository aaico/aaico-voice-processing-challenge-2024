import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2CTCTokenizer
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
import librosa  # Add this import

def load_labeled_data():
    # Load your labeled dataset (replace this with your actual loading mechanism)
    with open('label_samples.pkl', 'rb') as file:
        labeled_samples = pickle.load(file)

    # Assuming each sample is a 1D array, reshape it to 2D
    features = [np.expand_dims(sample[0], axis=1) for sample in labeled_samples]
    labels = np.array([sample[1] for sample in labeled_samples])

    return features, labels

def build_model(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    features, labels = load_labeled_data()

    # Pad features to ensure a consistent shape
    max_length = max(len(sample) for sample in features)
    features = [np.pad(sample, ((0, max_length - len(sample)), (0, 0))) for sample in features]

    # Convert features and labels to TensorFlow tensors
    features = tf.convert_to_tensor(features, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)

    # Flatten the features
    features = tf.reshape(features, (features.shape[0], -1))

    # Split data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        features.numpy(),  # Convert TensorFlow tensor to NumPy array
        labels.numpy(),    # Convert TensorFlow tensor to NumPy array
        test_size=0.2,
        random_state=42
    )

    # Feature shape for the neural network
    input_shape = features.shape[1:]

    # Load Wav2Vec2 model and tokenizer for sequence classification
    model_name = "facebook/wav2vec2-base-960h"
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)  # Use Wav2Vec2CTCTokenizer
    wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

    # Extract features from the Wav2Vec2 model
    wav2vec2_features = [tokenizer(sample[0], return_tensors="pt", padding="max_length", truncation=True, max_length=max_length, stride=max_length, sampling_rate=16000) for sample in labeled_samples]
    wav2vec2_features = [item['input_values'].numpy() for item in wav2vec2_features]

    # Convert Wav2Vec2 features to TensorFlow tensor
    wav2vec2_features = tf.convert_to_tensor(wav2vec2_features, dtype=tf.float32)

    # Flatten the Wav2Vec2 features
    wav2vec2_features = tf.reshape(wav2vec2_features, (wav2vec2_features.shape[0], -1))

    # Concatenate the original features with Wav2Vec2 features
    combined_features = tf.concat([features, wav2vec2_features], axis=1)

    # Build and compile the model
    model = build_model(combined_features.shape[1:])

    # Define callbacks (early stopping and model checkpoint)
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]

    # Train the model
    history = model.fit(
        combined_features.numpy(), labels.numpy(),
        validation_data=(x_val, y_val),
        epochs=20,  # Adjust as needed
        batch_size=32,  # Adjust as needed
        callbacks=callbacks
    )

    # Save the trained model
    model.save('final_model.h5')

    # Save training history for analysis or plotting
    with open('training_history.pkl', 'wb') as file:
        pickle.dump(history.history)

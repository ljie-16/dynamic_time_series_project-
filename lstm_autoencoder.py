import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Bidirectional
from keras.callbacks import EarlyStopping

def build_lstm_autoencoder(input_shape):
    """
    Builds an LSTM autoencoder model.
    """
    encoder_input = Input(shape=input_shape)
    encoded = Bidirectional(LSTM(50, activation='relu'))(encoder_input)
    decoded = RepeatVector(input_shape[0])(encoded)
    decoded = Bidirectional(LSTM(50, activation='relu', return_sequences=True))(decoded)
    decoded = TimeDistributed(Dense(input_shape[1]))(decoded)

    autoencoder = Model(encoder_input, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

def train_autoencoder(autoencoder, X_train):
    """
    Trains the LSTM autoencoder on the training data.
    """
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    autoencoder.fit(X_train, X_train, epochs=100, batch_size=64, validation_split=0.1, callbacks=[early_stop])
    return autoencoder

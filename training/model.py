import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout

def create_nn_model(input_dim, units_1, units_2, learning_rate):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units_1, activation='relu', name='hidden_layer_1'),
        Dropout(0.2),
        Dense(units_2, activation='relu', name='hidden_layer_2'),
        Dropout(0.2),
        Dense(1, activation='sigmoid', name='output_layer') # Binary classification
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    return model

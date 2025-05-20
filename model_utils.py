from tensorflow import keras
import tensorflow as tf

def build_modelo(conv_base, shape, n_classes=1):
    conv_base.trainable = False

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=shape),
        conv_base,
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(n_classes, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
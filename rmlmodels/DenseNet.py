import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, ReLU, Dropout, Softmax, Conv2D, 
    MaxPool2D, Add, concatenate, Activation, Flatten
)
# CuDNNGRU yerine standart GRU kullanılır, TF arka planda GPU varsa otomatik optimize eder
from tensorflow.keras.layers import Bidirectional, GRU 

def DenseNet(weights=None,
             input_shape=[2, 128],
             classes=11,
             **kwargs):
    
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')
    
    dr = 0.6
    # Giriş katmanı: [2, 128, 1] formatında bekler
    img_input = Input(shape=input_shape + [1], name='input')
    
    # Dense Block Yapısı
    x = Conv2D(256, (1, 3), activation="relu", name="conv1", padding='same')(img_input)
    
    x1 = Conv2D(256, (2, 3), name="conv2", padding='same')(x)
    
    # x ve x1 birleştiriliyor (Skip Connection/Dense Connection)
    x2 = concatenate([x, x1])
    x2 = Activation('relu')(x2)
    
    x3 = Conv2D(80, (1, 3), name="conv3", padding='same')(x2)
    
    # x, x1 ve x3 birleştiriliyor
    x4 = concatenate([x, x1, x3])
    x4 = Activation('relu')(x4)
    
    x = Conv2D(80, (1, 3), activation="relu", name="conv4", padding='same')(x4)
    x = Dropout(dr)(x)
    x = Flatten()(x)

    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(dr)(x)
    x = Dense(classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=img_input, outputs=x)

    # Ağırlıkları yükle
    if weights is not None:
        model.load_weights(weights)

    return model

# Test amaçlı çalıştırma kısmı
if __name__ == '__main__':
    from tensorflow import keras
    model = DenseNet(None, input_shape=[2, 128], classes=11)
    
    # Keras 3 uyumlu optimizer (lr -> learning_rate)
    adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    
    model.summary()

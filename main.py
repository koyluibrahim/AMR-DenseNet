import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
import sys

# Keras 3 ve TF 2.x için ortam ayarları
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers

# Modül bağımlılıklarını Colab'e tanıtmak için
import mltools, dataset2016
import rmlmodels.DenseNet as mcl

# Veriyi yükle
(mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx) = \
    dataset2016.load_data()

# Boyutları genişlet (Keras 3 için gerekli)
X_train = np.expand_dims(X_train, axis=3).astype('float32')
X_test = np.expand_dims(X_test, axis=3).astype('float32')
X_val = np.expand_dims(X_val, axis=3).astype('float32')

print(f"Eğitim verisi şekli: {X_train.shape}")
classes = mods

# Klasörleri oluştur (Hata almamak için)
os.makedirs('weights', exist_ok=True)
os.makedirs('figure', exist_ok=True)
os.makedirs('predictresult', exist_ok=True)

# Model Parametreleri
nb_epoch = 100 
batch_size = 400 

# Model Kurulumu
model = mcl.DenseNet()

# KRİTİK DEĞİŞİKLİK: 'adam' yerine güncel Optimizer nesnesi ve learning_rate
model.compile(
    loss='categorical_crossentropy', 
    metrics=['accuracy'], 
    optimizer=optimizers.Adam(learning_rate=0.001)
)

model.summary()

filepath = 'weights/weights.h5'

# Eğitim
history = model.fit(
    X_train, Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=(X_val, Y_val),
    callbacks=[
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        # 'patince' yazım hatası 'patience' olarak düzeltildi
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=5, min_lr=0.0000001),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
    ]
)

# Sonuçları Göster
mltools.show_history(history)

# Tahmin ve Görselleştirme Fonksiyonu (Predict)
def predict_and_plot(model):
    model.load_weights(filepath)
    test_Y_hat = model.predict(X_test, batch_size=batch_size)
    
    # mltools içindeki fonksiyonun parametre sırasına göre kontrol et
    confnorm, _, _ = mltools.calculate_confusion_matrix(Y_test, test_Y_hat, classes)
    
    # Toplam Karmaşıklık Matrisi
    mltools.plot_confusion_matrix(confnorm, labels=classes, save_filename='figure/total_confusion')

    acc = {}
    acc_mod_snr = np.zeros((len(classes), len(snrs)))
    
    for i, snr in enumerate(snrs):
        # SNR bazlı filtreleme
        test_SNRs = [lbl[x][1] for x in test_idx]
        idx_i = np.where(np.array(test_SNRs) == snr)
        test_X_i = X_test[idx_i]
        test_Y_i = Y_test[idx_i]

        test_Y_i_hat = model.predict(test_X_i)
        confnorm_i, cor, ncor = mltools.calculate_confusion_matrix(test_Y_i, test_Y_i_hat, classes)
        
        acc[snr] = 1.0 * cor / (cor + ncor)
        acc_mod_snr[:, i] = np.round(np.diag(confnorm_i)/np.sum(confnorm_i, axis=1), 3)
        
        print(f"SNR {snr}dB için Doğruluk: {acc[snr]:.4f}")

    # SNR vs Accuracy Grafiği
    plt.figure(figsize=(10, 6))
    plt.plot(snrs, [acc[x] for x in snrs], marker='o')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Doğruluk")
    plt.title("SNR Bazlı Sınıflandırma Başarısı")
    plt.grid(True)
    plt.savefig('figure/overall_acc_vs_snr.png')
    plt.show()

# Fonksiyonu çalıştır
predict_and_plot(model)

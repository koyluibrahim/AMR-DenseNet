import os, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import tensorflow as tf

import dataset2016
import rmlmodels.DenseNet as mcl

def run_evaluation(base_path='/content/drive/MyDrive/AMR_DenseNet_Projesi'):
    """
    Modeli yükler, test verisi üzerinde tahmin yapar ve detaylı metrikleri kaydeder.
    """
    # 1. Veriyi Yükle
    print("Veri seti yükleniyor...")
    (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx) = \
        dataset2016.load_data()
    
    X_test_4d = np.expand_dims(X_test, axis=3).astype('float32')
    classes = mods

    # 2. Modeli Kur ve Ağırlıkları Yükle
    print("Model hazırlanıyor...")
    model = mcl.DenseNet()
    
    weight_path = os.path.join(base_path, 'weights/weights.keras')
    if os.path.exists(weight_path):
        model.load_weights(weight_path)
        print(f"Ağırlıklar yüklendi: {weight_path}")
    else:
        print(f"HATA: Ağırlık dosyası bulunamadı: {weight_path}")
        return

    # 3. Tahmin ve Raporlama
    print("Tahminler yapılıyor...")
    test_Y_hat = model.predict(X_test_4d, batch_size=400)
    y_true = np.argmax(Y_test, axis=1)
    y_pred = np.argmax(test_Y_hat, axis=1)

    report_text = classification_report(y_true, y_pred, target_names=classes)
    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    print("\n--- Sınıflandırma Raporu ---")
    print(report_text)

    # 4. Kayıt İşlemleri
    save_dir = os.path.join(base_path, 'predictresult')
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'final_metrics_report.txt'), 'w') as f:
        f.write(report_text)
    
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(os.path.join(save_dir, 'final_metrics_table.csv'))
    print(f"\nSonuçlar {save_dir} klasörüne kaydedildi.")

if __name__ == "__main__":
    run_evaluation()

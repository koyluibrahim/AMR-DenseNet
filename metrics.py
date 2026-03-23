import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import tensorflow as tf

import dataset2016
import mltools
import rmlmodels.DenseNet as mcl

def run_evaluation(base_path='/content/drive/MyDrive/AMR_DenseNet_Projesi'):
    """
    Modeli yükler, SNR bazlı analiz yapar ve hem tablo hem grafik olarak sonuçları kaydeder.
    """
    # 1. Veri Setinin Yüklenmesi
    print("Veri seti yükleniyor...")
    (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx) = \
        dataset2016.load_data()
    
    X_test_4d = np.expand_dims(X_test, axis=3).astype('float32')
    classes = mods
    sorted_snrs = sorted(snrs)

    # 2. Model Mimarisi ve Ağırlıkların Yüklenmesi
    print("Model hazırlanıyor...")
    model = mcl.DenseNet()
    
    weight_path = os.path.join(base_path, 'weights/weights.keras')
    if os.path.exists(weight_path):
        model.load_weights(weight_path)
        print(f"Ağırlıklar başarıyla yüklendi: {weight_path}")
    else:
        print(f"HATA: Ağırlık dosyası bulunamadı: {weight_path}")
        return

    # 3. Genel Performans Analizi (Classification Report)
    print("Genel tahminler yapılıyor...")
    test_Y_hat = model.predict(X_test_4d, batch_size=400)
    y_true = np.argmax(Y_test, axis=1)
    y_pred = np.argmax(test_Y_hat, axis=1)

    report_text = classification_report(y_true, y_pred, target_names=classes)
    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    print("\n--- Genel Sınıflandırma Raporu ---")
    print(report_text)

    # 4. SNR Bazlı Performans Analizi (Accuracy vs SNR)
    print("\nSNR bazlı analiz başlatılıyor...")
    overall_accs = []
    
    for snr in sorted_snrs:
        # Mevcut SNR'a ait verileri filtrele
        test_SNRs = [lbl[x][1] for x in test_idx]
        idx_i = np.where(np.array(test_SNRs) == snr)
        test_X_i = X_test_4d[idx_i]
        test_Y_i = Y_test[idx_i]

        # Tahmin ve doğruluk hesaplama
        test_Y_i_hat = model.predict(test_X_i, verbose=0)
        confnorm_i, cor, ncor = mltools.calculate_confusion_matrix(test_Y_i, test_Y_i_hat, classes)
        
        current_acc = 1.0 * cor / (cor + ncor)
        overall_accs.append(current_acc)
        print(f"SNR: {snr}dB -> Doğruluk: {current_acc:.4f}")

    # 5. Grafik Çizimi ve Kayıt
    save_dir_figs = os.path.join(base_path, 'figure')
    save_dir_results = os.path.join(base_path, 'predictresult')
    os.makedirs(save_dir_figs, exist_ok=True)
    os.makedirs(save_dir_results, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_snrs, overall_accs, 'b-o', linewidth=2, markersize=8, label='Overall Accuracy')
    for x, y in zip(sorted_snrs, overall_accs):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=9)

    plt.title('Overall Classification Accuracy vs SNR')
    plt.xlabel('SNR [dB]')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    # Grafiği Kaydet
    plt.savefig(os.path.join(save_dir_figs, 'overall_acc_vs_snr.png'))
    plt.show()

    # Raporları Kaydet
    with open(os.path.join(save_dir_results, 'final_metrics_report.txt'), 'w') as f:
        f.write(report_text)
    
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(os.path.join(save_dir_results, 'final_metrics_table.csv'))
    
    print(f"\nAnaliz tamamlandı. Tüm sonuçlar '{base_path}' altına kaydedildi.")

if __name__ == "__main__":
    run_evaluation()

import pickle
import numpy as np
import os

def load_data(filename='RML2016.10a_dict.pkl'):
    # Colab veya yerel dizinde dosyanın varlığını kontrol et
    if not os.path.exists(filename):
        # Eğer dosya mevcut değilse Google Drive yolunu dene veya hata ver
        drive_path = '/content/drive/MyDrive/RML2016.10a_dict.pkl'
        if os.path.exists(drive_path):
            filename = drive_path
        else:
            raise FileNotFoundError(f"{filename} bulunamadı! Lütfen veri setini Colab'e yükleyin.")

    # Veriyi yükle
    with open(filename, 'rb') as f:
        Xd = pickle.load(f, encoding='iso-8859-1')
    
    mods, snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0, 1]] 
    X = []
    lbl = []
    train_idx = []
    val_idx = []
    
    np.random.seed(2016)
    a = 0

    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
            
            # İndeksleri belirle (600 eğitim, 200 doğrulama, 200 test - Toplam 1000 örnek/SNR)
            current_range = range(a * 1000, (a + 1) * 1000)
            train_samples = list(np.random.choice(current_range, size=600, replace=False))
            train_idx += train_samples
            
            remaining = list(set(current_range) - set(train_samples))
            val_samples = list(np.random.choice(remaining, size=200, replace=False))
            val_idx += val_samples
            a += 1

    X = np.vstack(X)
    n_examples = X.shape[0]
    test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))
    
    # Karıştır
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    def to_onehot(yy):
        yy1 = np.zeros([len(yy), len(mods)])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_val = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), val_idx)))
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
    
    return (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx)

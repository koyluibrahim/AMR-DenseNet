import matplotlib.pyplot as plt 
import numpy as np
import pickle
import os

def show_history(history, save_path='figure'):
    """
    Eğitim geçmişini görselleştirir ve belirtilen dizine kaydeder.
    save_path: Drive yolun (örn: '/content/drive/MyDrive/AMR_Proje_Sonuclar/figure')
    """
    os.makedirs(save_path, exist_ok=True)
    
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
    
    # Loss Grafiği
    plt.figure()
    plt.title('Training loss performance')
    plt.plot(history.epoch, history.history['loss'], label='train loss')
    plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'total_loss.png'))
    plt.show() 
    plt.close()

    # Accuracy Grafiği
    plt.figure()
    plt.title('Training accuracy performance')
    plt.plot(history.epoch, history.history[acc_key], label='train_acc')
    plt.plot(history.epoch, history.history[val_acc_key], label='val_acc')
    plt.legend()    
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'total_acc.png'))
    plt.show()
    plt.close()

    # Sayısal verileri kaydet (Grafiklerin yanındaki txt dosyaları için)
    np.savetxt(os.path.join(save_path, 'train_acc.txt'), np.array(history.history[acc_key]))
    np.savetxt(os.path.join(save_path, 'val_acc.txt'), np.array(history.history[val_acc_key]))
    np.savetxt(os.path.join(save_path, 'train_loss.txt'), np.array(history.history['loss']))
    np.savetxt(os.path.join(save_path, 'val_loss.txt'), np.array(history.history['val_loss']))

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.get_cmap("Blues"), labels=[], save_filename=None):
    """
    Karmaşıklık matrisini çizer. save_filename tam yol olmalıdır.
    """
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(cm*100, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    for i in range(len(tick_marks)):
        for j in range(len(tick_marks)):
            val = int(np.around(cm[i, j]*100))
            color = "white" if cm[i, j] > 0.5 else "black"
            plt.text(j, i, val, ha="center", va="center", color=color)

    plt.tight_layout()
    if save_filename:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        plt.savefig(save_filename)
    plt.show()
    plt.close()

def calculate_confusion_matrix(Y, Y_hat, classes):
    n_classes = len(classes)
    conf = np.zeros([n_classes, n_classes])
    
    for k in range(0, Y.shape[0]):
        i = np.argmax(Y[k, :])
        j = np.argmax(Y_hat[k, :])
        conf[i, j] += 1

    confnorm = np.zeros([n_classes, n_classes])
    for i in range(0, n_classes):
        row_sum = np.sum(conf[i, :])
        if row_sum != 0:
            confnorm[i, :] = conf[i, :] / row_sum

    right = np.sum(np.diag(conf))
    wrong = np.sum(conf) - right
    return confnorm, right, wrong

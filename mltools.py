import matplotlib.pyplot as plt 
import numpy as np
import pickle
import os

def show_history(history):
    # Keras 3/TF 2.x'te 'acc' yerine 'accuracy' kullanılır
    # Hata almamak için her iki ihtimali de kontrol ediyoruz
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
    
    plt.figure()
    plt.title('Training loss performance')
    plt.plot(history.epoch, history.history['loss'], label='train loss')
    plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('figure/total_loss.png')
    plt.show() # Colab'de anlık görmek için plt.show() ekledim
    plt.close()

    plt.figure()
    plt.title('Training accuracy performance')
    plt.plot(history.epoch, history.history[acc_key], label='train_acc')
    plt.plot(history.epoch, history.history[val_acc_key], label='val_acc')
    plt.legend()    
    plt.grid(True)
    plt.savefig('figure/total_acc.png')
    plt.show()
    plt.close()

    # Verileri kaydet
    np.savetxt('train_acc.txt', np.array(history.history[acc_key]))
    np.savetxt('val_acc.txt', np.array(history.history[val_acc_key]))
    np.savetxt('train_loss.txt', np.array(history.history['loss']))
    np.savetxt('val_loss.txt', np.array(history.history['val_loss']))

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.get_cmap("Blues"), labels=[], save_filename=None):
    plt.figure(figsize=(10, 8), dpi=100) # Colab için DPI değerini biraz düşürdüm, boyut arttı
    plt.imshow(cm*100, interpolation='nearest', cmap=cmap)
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
        # Klasörün varlığından emin ol
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        plt.savefig(save_filename)
    plt.show()
    plt.close()

def calculate_confusion_matrix(Y, Y_hat, classes):
    n_classes = len(classes)
    conf = np.zeros([n_classes, n_classes])
    
    # One-hot encoded Y'yi indexe çevir
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

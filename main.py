import os, random, sys, pickle, csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers


# Define global paths for persistent storage
base_path = '/content/drive/MyDrive/AMR_DenseNet_Projesi'
os.makedirs(f"{base_path}/weights", exist_ok=True)
os.makedirs(f"{base_path}/figure", exist_ok=True)
os.makedirs(f"{base_path}/predictresult", exist_ok=True)

# Ensure local directories exist for temporary runtime safety
os.makedirs('weights', exist_ok=True)
os.makedirs('figure', exist_ok=True)
os.makedirs('predictresult', exist_ok=True)

# Module imports for data loading and visualization
import mltools, dataset2016
import rmlmodels.DenseNet as mcl

# --- Dataset Loading and Tensor Formatting ---
# Load RML2016.10a dataset and split indices
(mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx) = \
    dataset2016.load_data()

# Expand dimensions for 4D input [Samples, 2, 128, 1] as required by Conv2D layers
X_train = np.expand_dims(X_train, axis=3).astype('float32')
X_val = np.expand_dims(X_val, axis=3).astype('float32')
X_test = np.expand_dims(X_test, axis=3).astype('float32')

print(f"Dataset summary: Training shape {X_train.shape}")
classes = mods

# --- Model Initialization and Compilation ---
nb_epoch = 100 
batch_size = 400 

# Instantiate DenseNet architecture from the functional API
model = mcl.DenseNet()

# Compile model with Adam optimizer (Keras 3 compliant learning_rate)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], 
              optimizer=optimizers.Adam(learning_rate=0.001))
model.summary()

# --- Model Training Phase ---
# Checkpoint path for saving the best weights in native Keras format
filepath = f"{base_path}/weights/weights.keras"

history = model.fit(X_train, Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=(X_val, Y_val),
    callbacks = [
        # Save best model weights based on validation loss
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        # Dynamically adjust learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=5, min_lr=0.0000001),
        # Terminate training if validation loss does not improve for 50 epochs
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
    ]
)

# Visualize and export training history (Loss/Accuracy curves)
mltools.show_history(history)

# --- Post-Training Analysis and Metrics ---
def predict(model):
    # Load the optimized weights from persistent storage
    model.load_weights(filepath)
    
    # Calculate overall categorical accuracy on test set
    score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
    print(f"Overall Test Score: {score}")

    # Generate predictions for full test set
    test_Y_hat = model.predict(X_test, batch_size=batch_size)
    
    # Calculate and plot global normalized confusion matrix
    confnorm, _, _ = mltools.calculate_confusion_matrix(Y_test, test_Y_hat, classes)
    mltools.plot_confusion_matrix(confnorm, labels=classes, 
                                  save_filename=f'{base_path}/figure/total_confusion.png')

    acc = {}
    acc_mod_snr = np.zeros((len(classes), len(snrs)))
    
    # Iterate through each SNR level for granular performance analysis
    for i, snr in enumerate(snrs):
        # Filter test data by current SNR
        test_SNRs = [lbl[x][1] for x in test_idx]
        idx_i = np.where(np.array(test_SNRs) == snr)
        test_X_i = X_test[idx_i]
        test_Y_i = Y_test[idx_i]

        # Prediction and matrix calculation per SNR
        test_Y_i_hat = model.predict(test_X_i)
        confnorm_i, cor, ncor = mltools.calculate_confusion_matrix(test_Y_i, test_Y_i_hat, classes)
        
        # Store accuracy results per SNR
        acc[snr] = 1.0 * cor / (cor + ncor)
        result = cor / (cor + ncor)
        
        # Log results to CSV format for external analysis
        with open(f'{base_path}/predictresult/acc_log.csv', 'a', newline='') as f0:
            write0 = csv.writer(f0)
            write0.writerow([snr, result])
            
        # Plot individual confusion matrices for each SNR level
        mltools.plot_confusion_matrix(confnorm_i, labels=classes, title=f"Confusion Matrix (SNR={snr})",
                                      save_filename=f"{base_path}/figure/Confusion_SNR_{snr}.png")

        # Extract diagonal values for individual modulation accuracy per SNR
        acc_mod_snr[:, i] = np.round(np.diag(confnorm_i)/np.sum(confnorm_i, axis=1), 3)

    # Plot Accuracy vs SNR curves for all modulation types
    plt.figure(figsize=(12, 10))
    for j in range(len(classes)):
        plt.plot(snrs, acc_mod_snr[j], label=classes[j])
        for x, y in zip(snrs, acc_mod_snr[j]):
            plt.text(x, y, y, ha='center', va='bottom', fontsize=8)
    
    plt.legend()
    plt.grid()
    plt.title("Classification Accuracy for Each Modulation Type")
    plt.savefig(f'{base_path}/figure/acc_for_all_mods.png')
    plt.close()

    # Serialize and export accuracy dictionaries for future plotting
    with open(f'{base_path}/predictresult/acc_for_mod.dat', 'wb') as fd:
        pickle.dump(acc_mod_snr, fd)
    with open(f'{base_path}/predictresult/acc.dat', 'wb') as fd:
        pickle.dump(acc, fd)
    
    print(f"Analytical results exported to {base_path}")

# Execute prediction and analysis pipeline
predict(model)

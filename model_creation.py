import numpy as np
import tensorflow as tf
print(tf.__version__) 
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import json
from pathlib import Path


DATA_DIR = "CSPProject/CSP-Project-Ising-CNN/data"

BATCH_SIZE = 32

def build_model(config_size, hidden_nodes):
    l2 = .05  # regularization strength 
    initializer = tf.keras.initializers.GlorotNormal()
    x = tf.keras.Input((config_size,))
    y = tf.keras.layers.Dense(hidden_nodes, activation='sigmoid', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    z = tf.keras.layers.Dense(2, activation='sigmoid')(y)
    model = tf.keras.Model(inputs=x, outputs=z)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()
    return model

for L in [10, 20, 30, 40, 60]:

    data = np.load(f"CSPProject/CSP-Project-Ising-CNN/data_uploaded/L{L}_ising.npz")
    """split data into input and output"""
    T = data["temperatures"]
    T_c = 2 / np.log(1 + np.sqrt(2))        
    labels = np.transpose(np.array([(T > T_c).astype(int), (T < T_c).astype(int)]))    #create labels from temperature
    configs = data["spins"]

    rng = np.random.default_rng(seed=42)
    idx = np.arange(len(T))
    rng.shuffle(idx)

    # permutation
    T = T[idx]
    configs = configs[idx]
    labels  = labels[idx, :]
    print(labels.shape)

    """split into training, validation and test data"""
    train_conf, val_conf, test_conf = np.split(configs, [80000, 90000])
    train_label, val_label, test_label = np.split(labels, [80000, 90000], axis=0)
    train_T, val_T, test_T = np.split(T, [80000, 90000])
    #print(train_conf.shape)

    model3_2 = build_model(configs.shape[1], 100)

    w_init, b_init = model3_2.layers[1].get_weights()

    history = model3_2.fit(
        train_conf,
        train_label,
        validation_data = (val_conf, val_label),
        batch_size = 256,
        epochs = 75
    )


    file_path = f'CSPProject/CSP-Project-Ising-CNN/training_history_100_l2=.05/L{L}.json'
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(history.history, f)

    print(f"Successfully saved history to {file_path}")

    model3_2.save(f"CSPProject/CSP-Project-Ising-CNN/models_100_l2=.05/ising_classifier_L{L}.h5")



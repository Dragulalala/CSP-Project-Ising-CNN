import numpy as np
import tensorflow as tf
print(tf.__version__) 
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

DATA_DIR = "CSPProject/CSP-Project-Ising-CNN/data"

BATCH_SIZE = 32

def build_model(config_size, hidden_nodes):
    l2 = 1e-4  # regularization strength λ
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
    x = tf.keras.Input((config_size,))
    y = tf.keras.layers.Dense(hidden_nodes, activation='sigmoid', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    z = tf.keras.layers.Dense(2, activation='sigmoid')(y)
    model = tf.keras.Model(inputs=x, outputs=z)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()
    return model

for L in [10, 20, 30, 40, 60]:

    data = np.load(f"CSPProject/CSP-Project-Ising-CNN/data_decorr/L{L}_ising.npz")
    """split data into input and output"""
    T = data["temperatures"]
    T_c = 2 / np.log(1 + np.sqrt(2))        
    labels = np.transpose(np.array([(T > T_c).astype(int), (T < T_c).astype(int)]))    #create labels from temperature
    configs = data["spins"]

    rng = np.random.default_rng(seed=42)
    idx = np.arange(len(T))
    rng.shuffle(idx)

    # Apply the same permutation to all arrays
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
        epochs = 200
    )

    # Save after training
    model3_2.save(f"CSPProject/CSP-Project-Ising-CNN/models_100/ising_classifier_L{L}.h5")



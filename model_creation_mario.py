import numpy as np
import tensorflow as tf
print(tf.__version__) 
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
from pathlib import Path

DATA_DIR = "data_decorr/"
OUT_DIR = "models_100_mario/"

BATCH_SIZE = 32
LAM_VAR = 3e-1   # weight-variance regularization strength (pushes w_ji toward being constant in i)
L2 = 4e-2        # keep overall weight magnitude bounded


@tf.keras.utils.register_keras_serializable()
class VarianceL2Regularizer(tf.keras.regularizers.Regularizer):
    #L2 plus a penalty on the variance of each neuron's weight vector across
    #input sites. Built from tf.math.reduce_variance and tf.reduce_sum — pushes
    #h_j toward being a function of the magnetization alone.
    def __init__(self, l2, lam_var):
        self.l2 = l2
        self.lam_var = lam_var

    def __call__(self, w):
        return (self.l2 * tf.reduce_sum(tf.square(w))
                + self.lam_var * tf.reduce_sum(tf.math.reduce_variance(w, axis=0)))

    def get_config(self):
        return {'l2': float(self.l2), 'lam_var': float(self.lam_var)}
"""

@tf.keras.utils.register_keras_serializable()
class WeightVarianceRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, lam_var):
        self.lam_var = lam_var
    def __call__(self, w):
        return self.lam_var * tf.reduce_sum(tf.math.reduce_variance(w, axis=0))
    def get_config(self):
        return {'lam_var': float(self.lam_var)}
"""
        
def build_model(config_size, hidden_nodes, l2):
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0/config_size, seed=42)

    x = tf.keras.Input((config_size,))
    
    y = tf.keras.layers.Dense(
        hidden_nodes,
        activation='sigmoid',
        kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2)
        #kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3.0, axis=0)
    )(x)
    z = tf.keras.layers.Dense(2, activation='softmax')(y)
    model = tf.keras.Model(inputs=x, outputs=z)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()
    return model


for L in [10, 20, 30, 40, 60]:

    data = np.load(f"{DATA_DIR}/L{L}_ising.npz")
    """split data into input and output"""
    T = data["temperatures"]
    T_c = 2 / np.log(1 + np.sqrt(2))
    labels = np.transpose(np.array([(T > T_c).astype(int), (T < T_c).astype(int)]))
    configs = data["spins"]

    rng = np.random.default_rng(seed=42)
    idx = np.arange(len(T))
    rng.shuffle(idx)

    # Apply the same permutation to all arrays
    T = T[idx]
    configs = configs[idx]
    labels = labels[idx, :]
    print(labels.shape)

    """split into training, validation and test data"""
    train_conf, val_conf = np.split(configs, [80000])
    train_label, val_label = np.split(labels, [80000], axis=0)
    train_T, val_T = np.split(T, [80000])

    model3_2 = build_model(configs.shape[1], 100, L2)

    w_init, b_init = model3_2.layers[1].get_weights()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=1e-6)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model3_2.fit(
        train_conf,
        train_label,
        validation_data=(val_conf, val_label),
        batch_size=BATCH_SIZE,
        epochs=100,
        callbacks=[reduce_lr, early_stop]
    )

    file_path = f'{OUT_DIR}/training_history/L{L}.json'
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(history.history, f)

    print(f"Successfully saved history to {file_path}")

    # Save after training
    model3_2.save(f"{OUT_DIR}/ising_classifier_L{L}.h5")
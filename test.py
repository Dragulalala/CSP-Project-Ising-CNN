import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

L = 10
T_c = 2 / np.log(1 + np.sqrt(2))
HIDDEN_N = 100

data_train = np.load(f"CSPProject/CSP-Project-Ising-CNN/data_uploaded/L{L}_ising.npz")
configs_all = data_train["spins"]
T_all = data_train["temperatures"]
labels_all = np.column_stack([(T_all > T_c).astype(int), (T_all < T_c).astype(int)])

idx = np.random.permutation(len(configs_all))
train_conf, val_conf, _ = np.split(configs_all[idx], [80000, 90000])
train_label, val_label, _ = np.split(labels_all[idx], [80000, 90000])

def build_model(l2_reg):
    initializer = tf.keras.initializers.GlorotNormal() 
    x = Input((train_conf.shape[1],))
    h = Dense(
        HIDDEN_N,
        activation='sigmoid',
        kernel_initializer=initializer,
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    out = Dense(2, activation='sigmoid')(h)
    model = Model(inputs=x, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

l2_values = np.linspace(.01, .1, 10)
results = []

print(f"{'L2 Value':<10} | {'Train Acc':<10} | {'Val Acc':<10}")
print("-" * 35)

for l2 in l2_values:
    model = build_model(l2)
    history = model.fit(
        train_conf, train_label,
        validation_data=(val_conf, val_label),
        batch_size=256,
        epochs=50,  # Increased slightly to allow regularization to converge
        verbose=0
    )
    
    val_acc = history.history['val_accuracy'][-1]
    train_acc = history.history['accuracy'][-1]
    results.append(val_acc)
    val_acc = model.evaluate(val_conf, val_label, verbose=0)[1]
    
    weights, biases = model.layers[1].get_weights()
    mean_std = np.mean(np.std(weights, axis=0))
    
    print(f"{l2:.1e}    | {val_acc:.4f}    | {mean_std:.6f}")

    print(f"{l2:.1e}    | {train_acc:.4f}    | {val_acc:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(l2_values, results, marker='o', color='teal', linewidth=2)
plt.xscale('log')
plt.title(f'Effect of L2 Regularization on Accuracy (L={L})', fontsize=14)
plt.xlabel('L2 Penalty ($\lambda$) - Log Scale', fontsize=12)
plt.ylabel('Validation Accuracy', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.legend()

plt.savefig('l2_tuning_results2.png', dpi=150)
plt.show()

best_l2 = l2_values[np.argmax(results)]
print(f"\nPeak accuracy achieved at L2 = {best_l2:.1e}")
print("Tip: If you want clear lines, choose the highest L2 that maintains high accuracy.")
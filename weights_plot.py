import numpy as np
import tensorflow as tf
print(tf.__version__)
from matplotlib import pyplot as plt
import pandas as pd

# plot pre-activation (hidden args) vs magnetization for several model sets
L = [10, 20, 30, 40, 60]

fig, axs = plt.subplots(1, 5, figsize=(20, 4))
for i, l in enumerate(L):
    new_model = tf.keras.models.load_model(f'CSPProject/CSP-Project-Ising-CNN/models_100_l2=.05/ising_classifier_L{l}.h5', compile=False)
    data = np.load(f'CSPProject/CSP-Project-Ising-CNN/data_test_4/L{l}_ising.npz')
    weights, bias = new_model.layers[1].get_weights()
    hidden_args = data['spins'] @ weights + bias
    m = np.mean(data['spins'], axis=1)
    for j in range(hidden_args.shape[1]):
        axs[i].scatter(m, hidden_args[:, j], s=1, alpha=0.3, color=plt.cm.viridis(j/weights.shape[1]))
    axs[i].set_title(f'L={l}')
    axs[i].set_xlabel('magnetization')
    axs[i].set_ylabel('hidden pre-activation')
plt.tight_layout()
#plt.savefig('CSPProject/CSP-Project-Ising-CNN/hidden_args_3_l2=.05.png')
print('Saved hidden_args_3_l2=.05.png')

fig, axs = plt.subplots(1, 5, figsize=(20, 4))
for i, l in enumerate(L):
    new_model = tf.keras.models.load_model(f'CSPProject/CSP-Project-Ising-CNN/models_100_/ising_classifier_L{l}.h5', compile=False)
    data = np.load(f'CSPProject/CSP-Project-Ising-CNN/data_test_4/L{l}_ising.npz')
    weights, bias = new_model.layers[1].get_weights()
    hidden_args = data['spins'] @ weights + bias
    m = np.mean(data['spins'], axis=1)
    for j in range(hidden_args.shape[1]):
        axs[i].scatter(m, hidden_args[:, j], s=1, alpha=0.3, color=plt.cm.viridis(j/weights.shape[1]))
    axs[i].set_title(f'L={l}')
    axs[i].set_xlabel('magnetization')
    axs[i].set_ylabel('hidden pre-activation')
plt.tight_layout()
#plt.savefig('CSPProject/CSP-Project-Ising-CNN/hidden_args_100.png')
print('Saved hidden_args_100.png')

fig, axs = plt.subplots(1, 5, figsize=(20, 4))
for i, l in enumerate(L):
    new_model = tf.keras.models.load_model(f'CSPProject/CSP-Project-Ising-CNN/models_100/ising_classifier_L{l}.h5', compile=False)
    data = np.load(f'CSPProject/CSP-Project-Ising-CNN/data_test_4/L{l}_ising.npz')
    weights, bias = new_model.layers[1].get_weights()
    hidden_args = data['spins'] @ weights + bias
    m = np.mean(data['spins'], axis=1)
    for j in range(hidden_args.shape[1]):
        axs[i].scatter(m, hidden_args[:, j], s=1, alpha=0.3, color=plt.cm.viridis(j/weights.shape[1]))
    axs[i].set_title(f'L={l}')
    axs[i].set_xlabel('magnetization')
    axs[i].set_ylabel('hidden pre-activation')
plt.tight_layout()
#plt.savefig('CSPProject/CSP-Project-Ising-CNN/hidden_args_100_op.png')
print('Saved hidden_args_100_op.png')

plt.figure(figsize=(10, 6))
new_model = tf.keras.models.load_model(f'CSPProject/CSP-Project-Ising-CNN/models_100_2/ising_classifier_L40.h5', compile=False)
data = np.load(f'CSPProject/CSP-Project-Ising-CNN/data_test_4/L40_ising.npz')
weights, bias = new_model.layers[1].get_weights()
hidden_args = data['spins'] @ weights + bias
m = np.mean(data['spins'], axis=1)
for j in range(hidden_args.shape[1]):
    plt.scatter(m, hidden_args[:, j], s=1, alpha=0.3, color=plt.cm.viridis(j/weights.shape[1]))
plt.title('L=40')
plt.xlabel('magnetization')
plt.ylabel('hidden pre-activation')
plt.tight_layout()
#plt.savefig('CSPProject/CSP-Project-Ising-CNN/hidden_args_L40.png')
print('Saved hidden_args_L40.png')

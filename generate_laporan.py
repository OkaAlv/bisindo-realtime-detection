import os
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import cv2

# --- KONFIGURASI FOLDER & FILE ---
folder_output = 'hasil_visualisasi'
os.makedirs(folder_output, exist_ok=True)

path_model = 'model_bisindo_statis.h5' 
# PENTING: Ubah ini ke salah satu file gambar dataset Anda!
path_gambar = 'dataset_statis_224\A\A_01_0016.jpg' # <-- SESUAIKAN NAMA FILENYA

print("Memuat Model dan Gambar...")
model = keras.models.load_model(path_model)
img = cv2.imread(path_gambar)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = np.expand_dims(img_rgb, axis=0)

# --- VISUALISASI 1: INPUT RGB ---
plt.figure(figsize=(5, 5))
plt.imshow(img_rgb)
plt.title('1. Input Gambar RGB (224x224x3)')
plt.axis('off')
plt.savefig(f'{folder_output}/02_input_rgb.png', bbox_inches='tight')
plt.close()

# --- MEMBONGKAR MODEL (MENCARI LAYER CONV2D) ---
layer_outputs = [layer.output for layer in model.layers if 'conv2d' in layer.name]
visualisasi_model = keras.models.Model(inputs=model.inputs, outputs=layer_outputs)
feature_maps = visualisasi_model.predict(img_tensor)

# --- VISUALISASI 2: FEATURE MAPS (CONV2D LAYER 1) ---
# Layer pertama biasanya paling jelas memperlihatkan bentuk garis tepi/siluet jari
layer1_features = feature_maps[0] 

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle('2. Hasil Feature Extraction (Lapisan Conv2D Pertama)', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < 16: # Kita ambil 16 filter pertama saja dari 32
        ax.imshow(layer1_features[0, :, :, i], cmap='viridis')
        ax.set_title(f'Filter {i+1}')
    ax.axis('off')

plt.tight_layout()
plt.savefig(f'{folder_output}/03_feature_maps_conv2d.png', bbox_inches='tight')
plt.close()

print(f"\nSelesai! Silakan cek folder '{folder_output}' di VS Code Anda.")
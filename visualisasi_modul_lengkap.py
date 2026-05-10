import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
import cv2

# --- 1. KONFIGURASI FOLDER & FILE ---
folder_output = 'laporan_tahapan_cnn'
os.makedirs(folder_output, exist_ok=True)

path_model = 'model_bisindo_statis.h5' 
# PENTING: Ganti dengan jalur gambar yang valid dari dataset Anda!
path_gambar = 'dataset_statis_224\A\A_01_0017.jpg' 

print("Memuat Model dan Gambar...")
model = models.load_model(path_model)

# Ekstrak daftar nama kelas (jika model tidak menyimpannya, kita set manual atau ambil dari folder)
try:
    # Coba ambil folder dari dataset untuk label plot terakhir
    class_names = sorted(os.listdir('dataset_statis_224'))
except:
    class_names = [f"Kelas_{i}" for i in range(model.layers[-1].output_shape[1])]

# Baca gambar
img_bgr = cv2.imread(path_gambar)
if img_bgr is None:
    raise ValueError(f"Gambar tidak ditemukan di {path_gambar}")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_tensor = np.expand_dims(img_rgb, axis=0) # Ubah ke format (1, 224, 224, 3)

# --- FUNGSI BANTUAN UNTUK MENGAMBIL OUTPUT LAYER ---
def get_layer_output(model, layer_index, data):
    # Membuat model sementara yang terpotong sampai layer tertentu
    temp_model = models.Model(inputs=model.inputs, outputs=model.layers[layer_index].output)
    return temp_model.predict(data)

def plot_feature_maps(f_maps, title, filename, max_filters=16):
    # Fungsi untuk menggambar hasil Conv2D / MaxPooling2D
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(title, fontsize=14)
    for i, ax in enumerate(axes.flat):
        if i < max_filters and i < f_maps.shape[-1]:
            ax.imshow(f_maps[0, :, :, i], cmap='viridis')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{folder_output}/{filename}')
    plt.close()

print("Memulai proses bedah model...")

# --- TAHAP 0: INPUT ORIGINAL ---
plt.figure(figsize=(5, 5))
plt.imshow(img_rgb)
plt.title(f'0. Input RGB Asli\nUkuran: {img_rgb.shape}')
plt.axis('off')
plt.savefig(f'{folder_output}/00_input_rgb.png')
plt.close()
print("- Tahap 0 (Input) diekstrak.")

# --- TAHAP 1: RESCALING (Normalisasi) ---
# Layer index 0
hasil_rescaling = get_layer_output(model, 0, img_tensor)
plt.figure(figsize=(5, 5))
plt.imshow(hasil_rescaling[0])
plt.title(f'1. Hasil Rescaling (Normalisasi)\nRentang Piksel: {np.min(hasil_rescaling):.2f} s/d {np.max(hasil_rescaling):.2f}')
plt.axis('off')
plt.savefig(f'{folder_output}/01_rescaling.png')
plt.close()
print("- Tahap 1 (Rescaling) diekstrak.")

# --- TAHAP 2: BLOCK 1 (Conv2D + MaxPool) ---
# Layer index 1 (Conv2D) dan 2 (MaxPool)
conv1_out = get_layer_output(model, 1, img_tensor)
pool1_out = get_layer_output(model, 2, img_tensor)
plot_feature_maps(conv1_out, f'2A. Block 1 - Conv2D (Ekstraksi Garis Tepi)\nUkuran: {conv1_out.shape}', '02a_block1_conv2d.png')
plot_feature_maps(pool1_out, f'2B. Block 1 - MaxPooling (Reduksi Resolusi)\nUkuran: {pool1_out.shape}', '02b_block1_maxpool.png')
print("- Tahap 2 (Block 1) diekstrak.")

# --- TAHAP 3: BLOCK 2 (Conv2D + MaxPool) ---
# Layer index 3 (Conv2D) dan 4 (MaxPool)
conv2_out = get_layer_output(model, 3, img_tensor)
pool2_out = get_layer_output(model, 4, img_tensor)
plot_feature_maps(conv2_out, f'3A. Block 2 - Conv2D (Pencarian Pola Kompleks)\nUkuran: {conv2_out.shape}', '03a_block2_conv2d.png')
plot_feature_maps(pool2_out, f'3B. Block 2 - MaxPooling\nUkuran: {pool2_out.shape}', '03b_block2_maxpool.png')
print("- Tahap 3 (Block 2) diekstrak.")

# --- TAHAP 4: BLOCK 3 (Conv2D + MaxPool) ---
# Layer index 5 (Conv2D) dan 6 (MaxPool)
conv3_out = get_layer_output(model, 5, img_tensor)
pool3_out = get_layer_output(model, 6, img_tensor)
plot_feature_maps(conv3_out, f'4A. Block 3 - Conv2D (Pola Abstrak/Mendalam)\nUkuran: {conv3_out.shape}', '04a_block3_conv2d.png')
plot_feature_maps(pool3_out, f'4B. Block 3 - MaxPooling\nUkuran: {pool3_out.shape}', '04b_block3_maxpool.png')
print("- Tahap 4 (Block 3) diekstrak.")

# --- TAHAP 5: FLATTEN (Menggepengkan Matriks) ---
# Layer index 7
flatten_out = get_layer_output(model, 7, img_tensor)
plt.figure(figsize=(12, 3))
# Kita visualisasikan vektor 1D sebagai semacam "Barcode" (Heatmap 1D)
plt.imshow(flatten_out, aspect='auto', cmap='plasma')
plt.title(f'5. Hasil Flatten (Matriks diubah jadi Vektor 1D)\nPanjang Vektor: {flatten_out.shape[1]} angka')
plt.xlabel('Indeks Array')
plt.yticks([])
plt.colorbar(label='Nilai Aktivasi')
plt.tight_layout()
plt.savefig(f'{folder_output}/05_flatten_barcode.png')
plt.close()
print("- Tahap 5 (Flatten) diekstrak.")

# --- TAHAP 6: DENSE LAYER (Hidden Layer) ---
# Layer index 8
dense_out = get_layer_output(model, 8, img_tensor)
plt.figure(figsize=(12, 4))
plt.bar(range(dense_out.shape[1]), dense_out[0], color='teal')
plt.title(f'6. Jaringan Saraf Tersembunyi (Dense Layer - 128 Neuron)\nMenghitung bobot akhir sebelum menebak')
plt.xlabel('Neuron ke-')
plt.ylabel('Nilai Aktivasi (ReLU)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'{folder_output}/06_dense_hidden.png')
plt.close()
print("- Tahap 6 (Dense Hidden) diekstrak.")

# --- TAHAP 7: OUTPUT PREDIKSI (Softmax) ---
# Layer index 10 (melewati Dropout di index 9)
output_pred = get_layer_output(model, 10, img_tensor)
plt.figure(figsize=(10, 5))
bars = plt.bar(class_names, output_pred[0] * 100, color='coral')
plt.title('7. Hasil Akhir Prediksi (Output Softmax)')
plt.xlabel('Kelas / Huruf')
plt.ylabel('Persentase Keyakinan (%)')
plt.ylim(0, 100)

# Menambahkan angka persentase di atas setiap batang grafik
for bar in bars:
    yval = bar.get_height()
    if yval > 1: # Hanya tampilkan angka yang cukup besar
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{folder_output}/07_output_prediksi.png')
plt.close()
print("- Tahap 7 (Output Klasifikasi) diekstrak.")

print(f"\nSELESAI! Buka folder '{folder_output}' untuk melihat hasil visualisasi step-by-step model Anda.")
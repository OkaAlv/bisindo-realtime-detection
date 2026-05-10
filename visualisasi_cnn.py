import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

# --- 1. KONFIGURASI ---
# Pastikan nama file model sesuai dengan yang Anda simpan sebelumnya
path_model = 'model_bisindo_statis.h5' 

# Gunakan satu gambar tangan yang SUDAH di-crop 224x224 dari folder dataset Anda
# (Ganti path ini sesuai dengan file yang ada di laptop Anda)
path_gambar_tes = 'dataset_statis_224/A/A_01_0000.jpg' 

print("Memuat model CNN...")
model = tf.keras.models.load_model(path_model)

# --- 2. MEMBUAT MODEL EKSTRAKTOR VISUAL ---
# Kita akan mengambil output HANYA dari layer konvolusi (Conv2D)
layer_outputs = [layer.output for layer in model.layers if 'conv2d' in layer.name]

if not layer_outputs:
    print("Error: Tidak menemukan layer Conv2D di dalam model Anda.")
    exit()

# Membuat "Model Mini" yang input-nya sama dengan model asli, 
# tapi output-nya berhenti di layer Conv2D untuk kita intip isinya
visualisasi_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs)

# --- 3. PERSIAPAN GAMBAR ---
print("Memproses gambar...")
img = cv2.imread(path_gambar_tes)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# CNN Keras meminta format data (Batch, Height, Width, Channel)
# Jadi kita tambahkan satu dimensi ekstra di depan menggunakan np.expand_dims
img_tensor = np.expand_dims(img_rgb, axis=0) 

# --- 4. PROSES EKSTRAKSI FITUR (INFERENCE) ---
# Memasukkan gambar ke "Model Mini"
feature_maps = visualisasi_model.predict(img_tensor)

# --- 5. MENGGAMBAR KANVAS VISUALISASI ---
# (Kita ambil hasil dari layer Conv2D yang PERTAMA saja agar mudah dijelaskan)
layer_pertama = feature_maps[0] 

# Layer pertama kita punya 32 filter (kedalaman), kita tampilkan 16 saja agar layar tidak penuh
jumlah_filter_ditampilkan = 16 

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle('Visualisasi Feature Extraction (Layer Conv2D Pertama)', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < jumlah_filter_ditampilkan:
        # Mengambil satu per satu "kacamata filter" dari model
        feature_image = layer_pertama[0, :, :, i]
        
        # Menampilkan gambar dengan mode abu-abu (seperti X-Ray)
        ax.imshow(feature_image, cmap='viridis')
        ax.set_title(f'Filter {i+1}')
    ax.axis('off')

plt.tight_layout()
plt.show()
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# --- 1. KONFIGURASI ---
path_model = 'model_bisindo_statis.h5'
dataset_dir = 'dataset_statis_224' # Folder gambar training Anda
batch_size = 32
img_height = 224
img_width = 224

print("Memuat Model CNN...")
model = tf.keras.models.load_model(path_model)

print(f"Memuat 20% Data Validasi dari '{dataset_dir}'...")
# Kita panggil HANYA subset "validation" (20%)
# WAJIB menggunakan seed=123 agar data yang diambil persis sama 
# dengan data validasi saat proses training kemarin.
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123, 
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True # <--- UBAH JADI TRUE AGAR SEMUA HURUF TERCAMPUR
)

class_names = val_ds.class_names
print(f"Kelas yang dievaluasi: {class_names}")

# --- 2. MELAKUKAN PREDIKSI ---
print("\nAI sedang mengerjakan soal validasi (Memprediksi gambar)...")
y_true = []
y_pred = []

for images, labels in val_ds:
    # Ambil kunci jawaban asli
    y_true.extend(labels.numpy())
    
    # Lakukan normalisasi pixel (jika model Anda tidak menggunakan layers.Rescaling di dalamnya)
    # Karena model kita SUDAH ada layers.Rescaling(1./255), kita bisa langsung predict
    prediksi_batch = model.predict(images, verbose=0)
    
    # Ambil indeks dengan persentase tertinggi
    tebakan_batch = np.argmax(prediksi_batch, axis=1)
    y_pred.extend(tebakan_batch)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# --- 3. MENGHITUNG METRIK EVALUASI ---
print("\n" + "="*50)
print("     HASIL EVALUASI MODEL (20% DATA VALIDASI)")
print("="*50)

akurasi_total = np.mean(y_true == y_pred) * 100
print(f"Total Gambar Validasi: {len(y_true)} gambar")
print(f"AKURASI TOTAL KESELURUHAN: {akurasi_total:.2f}%\n")

print("CLASSIFICATION REPORT (Precision, Recall, F1-Score):")
laporan = classification_report(y_true, y_pred, target_names=class_names)
print(laporan)

# --- 4. VISUALISASI CONFUSION MATRIX ---
print("Membuat visualisasi Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', # Menggunakan warna Oranye agar beda dengan video (Biru)
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Pengujian 20% Data Validasi (Statis)', fontsize=16)
plt.ylabel('Jawaban Asli (True Label)', fontsize=12)
plt.xlabel('Tebakan AI (Predicted Label)', fontsize=12)

# Menyimpan hasil
os.makedirs('hasil_visualisasi', exist_ok=True)
plt.savefig('hasil_visualisasi/09_confusion_matrix_validasi.png', bbox_inches='tight')
print("Selesai! Gambar Confusion Matrix disimpan di 'hasil_visualisasi/09_confusion_matrix_validasi.png'")

# Tampilkan ke layar
plt.show()
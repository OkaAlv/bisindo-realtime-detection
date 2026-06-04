import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# --- 1. PERSIAPAN DATASET (STRATEGI GABUNGAN DINAMIS) ---
dataset_dir = 'dataset_statis_224'
dir_data_baru = 'dataset_prime_224'
batch_size = 32
img_height = 224
img_width = 224

print("=== Memuat Dataset Lama ===")
train_ds_lama = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds_lama = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

print("\n=== Memuat Dataset Baru (Prime) ===")
train_ds_baru = tf.keras.utils.image_dataset_from_directory(
    dir_data_baru,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds_baru = tf.keras.utils.image_dataset_from_directory(
    dir_data_baru,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# --- PROSES PENGGABUNGAN (CONCATENATE) ---
print("\nMenggabungkan kedua dataset di memori...")
train_ds = train_ds_lama.concatenate(train_ds_baru)
val_ds = val_ds_lama.concatenate(val_ds_baru)

# Mengacak ulang data gabungan agar tercampur rata
train_ds = train_ds.shuffle(buffer_size=1000, seed=123)

# Optimasi performa loading data (sangat membantu karena data 2x lebih besar)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Menyimpan nama-nama kelas (A, B, C, dst)
class_names = train_ds_lama.class_names
num_classes = len(class_names)
print(f"\nPenggabungan sukses! Siap melatih model untuk {num_classes} kelas: {class_names}")


# --- 2. MEMBANGUN ARSITEKTUR CNN ---
model = models.Sequential([
    # Layer Preprocessing (Normalisasi pixel 0-255 menjadi 0-1)
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    # --- TAHAP FEATURE EXTRACTION ---
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # --- TAHAP KLASIFIKASI ---
    # Menggepengkan matriks gambar menjadi vektor 1D
    layers.Flatten(),
    
    # Hidden Layer (Jaringan Saraf)
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Mencegah AI sekadar "menghafal" (Overfitting)
    
    # Output Layer (Sesuai jumlah huruf/kelas)
    layers.Dense(num_classes, activation='softmax')
])


# --- 3. KOMPILASI & TRAINING ---
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Tampilkan ringkasan layer di terminal
model.summary()

print("\nMemulai proses training gabungan...")
epochs = 10 # Berapa kali AI membaca seluruh dataset
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)


# --- 4. SIMPAN MODEL ---
model.save('model_bisindo_statis_combined.h5')
print("\nModel berhasil disimpan dengan nama 'model_bisindo_statis_combined.h5'!")


# --- 5. VISUALISASI GRAFIK EPOCH ---
os.makedirs('hasil_visualisasi', exist_ok=True)

plt.figure(figsize=(12, 4))
# Grafik Akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Akurasi Training', marker='o')
plt.plot(history.history['val_accuracy'], label='Akurasi Validasi', marker='o')
plt.title('Grafik Pergerakan Akurasi (Epoch)')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True)

# Grafik Error (Loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss Training', marker='o')
plt.plot(history.history['val_loss'], label='Loss Validasi', marker='o')
plt.title('Grafik Penurunan Error (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('hasil_visualisasi/01_grafik_training.png')
print("Grafik training berhasil disimpan di folder 'hasil_visualisasi'!")
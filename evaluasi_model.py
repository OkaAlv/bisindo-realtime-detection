import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mediapipe as mp

# --- 1. KONFIGURASI PENTING ---
path_model = 'model_bisindo_statis.h5'
folder_testing = 'Testing' # Sesuai dengan nama folder Anda
padding = 20

print("Memuat Model CNN...")
model = tf.keras.models.load_model(path_model)

# Mengambil daftar kelas (A, B, C...) dari nama subfolder
class_names = sorted([d for d in os.listdir(folder_testing) if os.path.isdir(os.path.join(folder_testing, d))])
print(f"Kelas yang akan diuji: {class_names}")

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Array untuk menyimpan kunci jawaban dan tebakan
y_true = []
y_pred = []

print("\nMemulai proses evaluasi video (ini mungkin memakan waktu beberapa menit)...")

# --- 2. LOOPING MEMBACA SETIAP VIDEO ---
for kelas_asli in class_names:
    path_kelas = os.path.join(folder_testing, kelas_asli)
    
    for file_video in os.listdir(path_kelas):
        if file_video.lower().endswith(('.mp4', '.avi', '.mov')):
            path_video = os.path.join(path_kelas, file_video)
            print(f"-> Memproses video: {kelas_asli}/{file_video} ...")
            
            cap = cv2.VideoCapture(path_video)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break # Video selesai
                
                # Proses MediaPipe
                h, w, c = frame.shape
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                # Jika tangan ditemukan di frame tersebut
                if results.multi_hand_landmarks:
                    x_min, y_min = w, h
                    x_max = y_max = 0
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                            x, y = int(lm.x * w), int(lm.y * h)
                            x_min, y_min = min(x_min, x), min(y_min, y)
                            x_max, y_max = max(x_max, x), max(y_max, y)
                            
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)
                    
                    if x_max > x_min and y_max > y_min:
                        try:
                            # Potong (Crop) dan Resize gambar tangan
                            cropped_hand = image_rgb[y_min:y_max, x_min:x_max]
                            resized_hand = cv2.resize(cropped_hand, (224, 224))
                            img_tensor = np.expand_dims(resized_hand, axis=0)
                            
                            # Prediksi AI
                            prediksi = model.predict(img_tensor, verbose=0)
                            indeks_tebakan = np.argmax(prediksi[0])
                            huruf_tebakan = class_names[indeks_tebakan]
                            
                            # Catat ke dalam buku rapor
                            y_true.append(kelas_asli)
                            y_pred.append(huruf_tebakan)
                        except Exception as e:
                            # Abaikan jika terjadi error pemotongan di pinggir layar
                            pass
                            
            cap.release()

# --- 3. MENGHITUNG METRIK EVALUASI ---
print("\n" + "="*50)
print("             HASIL EVALUASI MODEL VIDEO")
print("="*50)

# Pastikan y_true dan y_pred diubah ke numpy array agar sklearn tidak bingung
y_true = np.array(y_true)
y_pred = np.array(y_pred)

if len(y_true) == 0:
    print("ERROR: Tidak ada tangan yang terdeteksi di seluruh video, atau folder kosong!")
    exit()

akurasi_total = np.mean(y_true == y_pred) * 100
print(f"Total Frame yang diuji: {len(y_true)} frame")
print(f"AKURASI TOTAL KESELURUHAN: {akurasi_total:.2f}%\n")

print("CLASSIFICATION REPORT (Precision, Recall, F1-Score):")
laporan = classification_report(y_true, y_pred, labels=class_names)
print(laporan)

# --- 4. VISUALISASI CONFUSION MATRIX ---
print("Membuat visualisasi Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred, labels=class_names)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Pengujian Video Frame-by-Frame', fontsize=16)
plt.ylabel('Jawaban Asli (Video Sebenarnya)', fontsize=12)
plt.xlabel('Tebakan AI (Prediksi Model)', fontsize=12)

os.makedirs('hasil_visualisasi', exist_ok=True)
plt.savefig('hasil_visualisasi/08_confusion_matrix_video.png', bbox_inches='tight')
print("Selesai! Gambar Confusion Matrix disimpan di 'hasil_visualisasi/08_confusion_matrix_video.png'")
# Tampilkan ke layar
plt.show()
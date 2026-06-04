import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# --- 1. PERSIAPAN MODEL & KELAS ---
print("Memuat model CNN...")
model = load_model('model_bisindo_statis_combined.h5')

# Mengambil daftar huruf otomatis dari nama folder agar urutannya persis seperti saat training
dataset_dir = 'dataset_statis_224'
try:
    class_names = sorted(os.listdir(dataset_dir))
    print(f"Siap mendeteksi: {class_names}")
except FileNotFoundError:
    print(f"Folder {dataset_dir} tidak ditemukan. Pastikan jalurnya benar!")
    exit()

# --- 2. PERSIAPAN MEDIAPIPE (Pendeteksi Tangan) ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Mengingat BISINDO, kita set max_num_hands=2
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# --- 3. MENYALAKAN WEBCAM ---
cap = cv2.VideoCapture(0) # Angka 0 biasanya untuk webcam bawaan laptop
padding = 20

print("\n=== KAMERA MENYALA ===")
print("Tekan tombol 'q' di keyboard untuk mematikan kamera.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Efek Cermin (Flip) agar pergerakan tangan tidak terbalik di layar
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # MediaPipe butuh format warna RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # Jika tangan terdeteksi...
    if results.multi_hand_landmarks:
        x_min, y_min = w, h
        x_max = y_max = 0
        
        # 1. Menggambar kerangka tangan & Membuat kotak pembungkus (Bounding Box)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)
        
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Gambar kotak hijau di sekitar tangan
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # --- 4. TAHAP PREDIKSI OLEH CNN ---
        if x_max > x_min and y_max > y_min:
            # Potong gambar tangan
            cropped_hand = image_rgb[y_min:y_max, x_min:x_max]
            # Sesuaikan dengan ukuran input AI (224x224)
            resized_hand = cv2.resize(cropped_hand, (224, 224))
            
            # Ubah jadi matriks 4D (Batch_size, 224, 224, 3)
            img_tensor = np.expand_dims(resized_hand, axis=0)
            
            # Suruh AI menebak!
            prediksi = model.predict(img_tensor, verbose=0)
            indeks_tertinggi = np.argmax(prediksi[0])
            nilai_kepercayaan = prediksi[0][indeks_tertinggi] * 100
            huruf_ditebak = class_names[indeks_tertinggi]
            
            # Tampilkan teks tebakan di atas kotak jika AI cukup yakin (> 50%)
            if nilai_kepercayaan > 50:
                teks_hasil = f"{huruf_ditebak} ({nilai_kepercayaan:.1f}%)"
                cv2.rectangle(frame, (x_min, y_min - 40), (x_max, y_min), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, teks_hasil, (x_min + 5, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                
    # Tampilkan antarmuka ke layar
    cv2.imshow('Deteksi BISINDO Real-time', frame)
    
    # Tombol darurat untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Matikan sistem
cap.release()
cv2.destroyAllWindows()
print("\nKamera dimatikan. Selesai!")
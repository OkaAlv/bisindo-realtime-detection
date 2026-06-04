import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

print("Memuat model CNN...")
model = load_model('model_bisindo_statis_combined.h5')

dataset_dir = 'dataset_statis_224'
try:
    class_names = sorted(os.listdir(dataset_dir))
except FileNotFoundError:
    print(f"Folder {dataset_dir} tidak ditemukan.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
padding = 20

# --- VARIABEL BARU UNTUK MERANGKAI KATA ---
kalimat_terbentuk = ""
huruf_sebelumnya = ""
hitung_frame = 0
cooldown = 0
batas_frame = 15 # Tahan pose selama 15 frame (~0.5 detik) untuk mengetik
batas_cooldown = 30 # Jeda 30 frame (~1 detik) sebelum bisa mengetik huruf baru

print("\n=== KAMERA MENYALA ===")
print("Tekan 'q' untuk keluar. Tekan 'c' untuk menghapus teks (Clear).")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        x_min, y_min = w, h
        x_max = y_max = 0
        
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
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        if x_max > x_min and y_max > y_min:
            cropped_hand = image_rgb[y_min:y_max, x_min:x_max]
            resized_hand = cv2.resize(cropped_hand, (224, 224))
            img_tensor = np.expand_dims(resized_hand, axis=0)
            
            prediksi = model.predict(img_tensor, verbose=0)
            indeks_tertinggi = np.argmax(prediksi[0])
            nilai_kepercayaan = prediksi[0][indeks_tertinggi] * 100
            huruf_ditebak = class_names[indeks_tertinggi]
            
            # Tampilkan tebakan instan di atas tangan
            if nilai_kepercayaan > 60: # Threshold akurasi dinaikkan agar lebih ketat
                teks_hasil = f"{huruf_ditebak} ({nilai_kepercayaan:.1f}%)"
                cv2.rectangle(frame, (x_min, y_min - 40), (x_max, y_min), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, teks_hasil, (x_min + 5, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                
                # --- LOGIKA PENGETIKAN KATA ---
                if cooldown == 0: # Jika tidak sedang masa jeda
                    if huruf_ditebak == huruf_sebelumnya:
                        hitung_frame += 1
                    else:
                        huruf_sebelumnya = huruf_ditebak
                        hitung_frame = 0
                    
                    # Jika posisi tangan ditahan cukup lama
                    if hitung_frame >= batas_frame:
                        kalimat_terbentuk += huruf_ditebak
                        hitung_frame = 0
                        cooldown = batas_cooldown # Aktifkan jeda istirahat
                
    # Kurangi waktu jeda setiap frame
    if cooldown > 0:
        cooldown -= 1
        
    # --- MENAMPILKAN KATA YANG DIRANGKAI DI LAYAR ---
    # Kotak hitam transparan di bawah layar sebagai papan ketik
    cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f"Kata: {kalimat_terbentuk}", (20, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
                
    cv2.imshow('Deteksi BISINDO Real-time', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'): # Tekan 'C' untuk mereset kata
        kalimat_terbentuk = ""

cap.release()
cv2.destroyAllWindows()
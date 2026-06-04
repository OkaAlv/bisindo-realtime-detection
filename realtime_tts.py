import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import pyttsx3
import threading
import time # PENTING: Tambahan library untuk logika Timer

# --- 0. INISIALISASI TEXT-TO-SPEECH (TTS) ---
print("Memuat mesin suara...")
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def ucapkan_teks(teks):
    engine.say(teks)
    engine.runAndWait()

# --- 1. PERSIAPAN MODEL & KELAS ---
print("Memuat model CNN Gabungan...")
model = load_model('model_bisindo_statis_combined.h5')

dataset_dir = 'dataset_statis_224'
try:
    class_names = sorted(os.listdir(dataset_dir))
    print(f"Siap mendeteksi: {class_names}")
except FileNotFoundError:
    print(f"Folder {dataset_dir} tidak ditemukan.")
    exit()

# --- 2. PERSIAPAN MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# --- 3. MENYALAKAN WEBCAM ---
cap = cv2.VideoCapture(0)
padding = 20

# --- VARIABEL LOGIKA MERANGKAI KATA ---
kata_terbentuk = ""
huruf_kandidat = ""
waktu_mulai_tahan = time.time()
waktu_tangan_terakhir_terlihat = time.time()
cooldown_ngetik = False

print("\n=== KAMERA MENYALA ===")
print("CARA PAKAI:")
print("- Tahan gerakan 1.5 detik untuk merangkai 1 huruf.")
print("- Turunkan tangan dari kamera selama 2.5 detik agar AI membacakan katanya.")
print("- Tekan 'q' untuk keluar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        # Reset timer tangan hilang karena tangan sedang terlihat
        waktu_tangan_terakhir_terlihat = time.time() 
        
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
        
        # --- TAHAP PREDIKSI ---
        if x_max > x_min and y_max > y_min:
            cropped_hand = image_rgb[y_min:y_max, x_min:x_max]
            resized_hand = cv2.resize(cropped_hand, (224, 224))
            img_tensor = np.expand_dims(resized_hand, axis=0)
            
            prediksi = model.predict(img_tensor, verbose=0)
            indeks_tertinggi = np.argmax(prediksi[0])
            nilai_kepercayaan = prediksi[0][indeks_tertinggi] * 100
            huruf_ditebak = class_names[indeks_tertinggi]
            
            if nilai_kepercayaan > 50:
                # Tampilkan info huruf yang sedang ditebak di atas kotak hijau
                cv2.rectangle(frame, (x_min, y_min - 40), (x_max, y_min), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, f"{huruf_ditebak} ({nilai_kepercayaan:.0f}%)", (x_min + 5, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                
                # --- LOGIKA PENGETIKAN HURUF (TAHAN 1.5 DETIK) ---
                if huruf_ditebak == huruf_kandidat:
                    # Jika huruf sudah ditahan lebih dari 1.5 detik dan belum masuk mode cooldown
                    if not cooldown_ngetik and (time.time() - waktu_mulai_tahan > 1.5):
                        kata_terbentuk += huruf_ditebak
                        cooldown_ngetik = True # Kunci agar huruf tidak tertik berkali-kali kayak mesin ketik rusak
                else:
                    # Jika bentuk tangan berubah ke huruf lain, reset timer
                    huruf_kandidat = huruf_ditebak
                    waktu_mulai_tahan = time.time()
                    cooldown_ngetik = False
                    
    else:
        # --- LOGIKA KETIKA TANGAN TIDAK ADA DI LAYAR ---
        lama_tidak_terlihat = time.time() - waktu_tangan_terakhir_terlihat
        
        # 1. Reset sistem ketik: Jika tangan diturunkan sebentar, buka kunci cooldown 
        # (Berguna jika ingin mengetik huruf yang sama 2x, misal 'A' lalu 'A' lagi pada kata 'SAAT')
        if lama_tidak_terlihat > 0.5:
            cooldown_ngetik = False
            huruf_kandidat = ""
            
        # 2. Eksekusi Suara: Jika tangan benar-benar turun selama 2.5 detik
        if lama_tidak_terlihat > 2.5 and len(kata_terbentuk) > 0:
            print(f"Membacakan kata: {kata_terbentuk}")
            threading.Thread(target=ucapkan_teks, args=(kata_terbentuk,)).start()
            kata_terbentuk = "" # Kosongkan layar untuk persiapan merangkai kata baru

    # --- MENAMPILKAN KATA YANG DIRANGKAI DI BAWAH LAYAR ---
    # Membuat blok hitam di bawah agar teks warna kuning terlihat sangat jelas
    cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f"Kata: {kata_terbentuk}", (20, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
                
    cv2.imshow('Penerjemah BISINDO Real-time', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nKamera dimatikan. Selesai!")
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# --- 1. KONFIGURASI FILE TESTING ---
path_model = 'model_bisindo_statis.h5'

# MASUKKAN PATH FILE GAMBAR ATAU VIDEO ANDA DI SINI:
# Contoh Gambar: 'dataset_statis_224/A/A_01_0000.jpg'
# Contoh Video: 'video_test_A.mp4'
path_file_test = 'Testing/B/01.mp4' 


# --- 2. PERSIAPAN SISTEM ---
print("Memuat model CNN...")
model = load_model(path_model)

dataset_dir = 'dataset_statis_224'
try:
    class_names = sorted(os.listdir(dataset_dir))
except FileNotFoundError:
    print(f"Folder {dataset_dir} tidak ditemukan.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# --- 3. FUNGSI INTI PREDIKSI ---
# Fungsi ini memotong kode agar rapi, menerima 1 frame (gambar), dan mengembalikan gambar dengan tebakan
def proses_dan_prediksi(frame):
    h, w, c = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        x_min, y_min = w, h
        x_max = y_max = 0
        padding = 20
        
        # Cari bounding box 2 tangan
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
        
        # Crop dan Prediksi
        if x_max > x_min and y_max > y_min:
            try:
                cropped_hand = image_rgb[y_min:y_max, x_min:x_max]
                resized_hand = cv2.resize(cropped_hand, (224, 224))
                img_tensor = np.expand_dims(resized_hand, axis=0)
                
                prediksi = model.predict(img_tensor, verbose=0)
                indeks_tertinggi = np.argmax(prediksi[0])
                nilai_kepercayaan = prediksi[0][indeks_tertinggi] * 100
                huruf_ditebak = class_names[indeks_tertinggi]
                
                # Tampilkan hasil
                teks_hasil = f"{huruf_ditebak} ({nilai_kepercayaan:.1f}%)"
                cv2.rectangle(frame, (x_min, y_min - 40), (x_max, y_min), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, teks_hasil, (x_min + 5, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            except Exception as e:
                # Abaikan error jika proses crop keluar dari batas layar
                pass
                
    return frame

# --- 4. EKSEKUSI BERDASARKAN FORMAT FILE ---
path_lower = path_file_test.lower()

if path_lower.endswith(('.png', '.jpg', '.jpeg')):
    print(f"\n>> Memproses Gambar: {path_file_test}")
    img = cv2.imread(path_file_test)
    
    if img is None:
        print("ERROR: Gambar tidak ditemukan! Cek kembali path-nya.")
    else:
        hasil_gambar = proses_dan_prediksi(img)
        cv2.imshow("Hasil Pengujian Gambar", hasil_gambar)
        print("Tekan tombol APAPUN pada keyboard Anda (saat jendela gambar aktif) untuk menutup program.")
        cv2.waitKey(0) # Program berhenti dan menunggu Anda menekan tombol apapun
        cv2.destroyAllWindows()

elif path_lower.endswith(('.mp4', '.avi', '.mov')):
    print(f"\n>> Memproses Video: {path_file_test}")
    cap = cv2.VideoCapture(path_file_test)
    
    if not cap.isOpened():
        print("ERROR: Video tidak ditemukan! Cek kembali path-nya.")
    else:
        print("Tekan huruf 'q' pada keyboard untuk menghentikan video.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Video selesai diputar.")
                break
                
            hasil_frame = proses_dan_prediksi(frame)
            cv2.imshow("Hasil Pengujian Video", hasil_frame)
            
            # Waktu tunggu 30ms agar pemutaran video tidak terlalu cepat/lambat
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
else:
    print("ERROR: Format file tidak didukung! Harap gunakan gambar (.jpg/.png) atau video (.mp4).")
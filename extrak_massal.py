import cv2
import mediapipe as mp
import os
import numpy as np

# --- KONFIGURASI FOLDER ---
sumber_video = 'Train' 
folder_output = 'dataset_statis_224'

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
# PERUBAHAN KRUSIAL: max_num_hands diubah jadi 2!
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
padding = 20

print("=== MEMULAI EKSTRAKSI DATASET STATIS (MODE 2 TANGAN) ===")

for huruf in sorted(os.listdir(sumber_video)):
    path_huruf = os.path.join(sumber_video, huruf)
    if not os.path.isdir(path_huruf): continue
        
    output_huruf_dir = os.path.join(folder_output, huruf)
    os.makedirs(output_huruf_dir, exist_ok=True)
    print(f"\n>> Memproses Folder Huruf: {huruf}")
    
    for nama_video in sorted(os.listdir(path_huruf)):
        if not nama_video.endswith('.mp4'): continue
            
        video_path = os.path.join(path_huruf, nama_video)
        cap = cv2.VideoCapture(video_path)
        
        saved_count = 0
        nama_file_tanpa_ext = nama_video.split('.')[0]
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            h, w, c = frame.shape
            
            # PERUBAHAN KRUSIAL: Looping untuk MENGGABUNGKAN koordinat kedua tangan
            if results.multi_hand_landmarks:
                x_min, y_min = w, h
                x_max = y_max = 0
                
                # Baca semua tangan yang ada di layar (bisa 1, bisa 2)
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
                    cropped_hand = frame[y_min:y_max, x_min:x_max]
                    resized_hand = cv2.resize(cropped_hand, (224, 224))
                    
                    img_name = os.path.join(output_huruf_dir, f"{huruf}_{nama_file_tanpa_ext}_{saved_count:04d}.jpg")
                    cv2.imwrite(img_name, resized_hand)
                    saved_count += 1
        
        cap.release()
        print(f"   - Video {nama_video} selesai. Mendapatkan {saved_count} gambar.")

cv2.destroyAllWindows()
print("\n=== EKSTRAKSI SELESAI SELURUHNYA! ===")
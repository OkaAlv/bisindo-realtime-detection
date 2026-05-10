import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# --- KONFIGURASI UNTUK DEMO ---
# Siapkan 1 gambar mentah (belum di-crop) untuk bahan demo presentasi
# Anda bisa mengambil salah satu frame dari folder Train Anda
gambar_demo = 'for_visual/A_005.png' # Pastikan letak dan nama filenya sesuai nanti

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Baca gambar mentah
frame = cv2.imread(gambar_demo)
if frame is None:
    print("Gambar tidak ditemukan! Cek path-nya.")
    exit()

# 1. TAHAP ORIGINAL (BGR)
frame_asli = frame.copy()

# 2. TAHAP KONVERSI RGB & DETEKSI (MediaPipe)
image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = hands.process(image_rgb)

# Buat salinan untuk menggambar Bounding Box
frame_bbox = image_rgb.copy()
h, w, c = frame.shape
padding = 20

cropped_hand = None
resized_224 = None

if results.multi_hand_landmarks:
    x_min, y_min = w, h
    x_max = y_max = 0
    
    # Visualisasi titik-titik tangan (Landmarks)
    for hand_landmarks in results.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame_bbox, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x), max(y_max, y)
            
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    # Gambar Kotak Merah (Bounding Box)
    cv2.rectangle(frame_bbox, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
    
    # 3. TAHAP CROP
    cropped_hand = image_rgb[y_min:y_max, x_min:x_max]
    
    # 4. TAHAP RESIZE (224x224)
    resized_224 = cv2.resize(cropped_hand, (224, 224))

# --- MEMBUAT KANVAS VISUALISASI UNTUK PRESENTASI ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Visualisasi Pipeline Preprocessing Computer Vision - BISINDO', fontsize=16)

# Plot 1: Original
axes[0].imshow(cv2.cvtColor(frame_asli, cv2.COLOR_BGR2RGB))
axes[0].set_title('1. Frame Original')
axes[0].axis('off')

# Plot 2: Deteksi Tangan & Bounding Box
axes[1].imshow(frame_bbox)
axes[1].set_title('2. Deteksi MediaPipe (Hybrid)')
axes[1].axis('off')

# Plot 3: Hasil Crop
if cropped_hand is not None:
    axes[2].imshow(cropped_hand)
    axes[2].set_title('3. Hasil Crop Area Tangan')
else:
    axes[2].set_title('3. Tidak ada tangan')
axes[2].axis('off')

# Plot 4: Resize 224x224 RGB
if resized_224 is not None:
    axes[3].imshow(resized_224)
    axes[3].set_title('4. Resize 224x224 RGB (Siap ke CNN)')
else:
    axes[3].set_title('4. -')
axes[3].axis('off')

plt.tight_layout()
plt.show()
# 🤟 SignVoice ID: Sistem Penerjemah Real-time BISINDO Berbasis Suara

Proyek *Computer Vision* ini dibangun menggunakan arsitektur *Pipeline 2-Tahap*: **MediaPipe** untuk ekstraksi *landmark* tangan (lokalisasi) dan **Convolutional Neural Network (CNN)** dengan TensorFlow/Keras untuk klasifikasi bentuk tangan secara *real-time*. 

Sistem ini dirancang sebagai **Aplikasi Penerjemah Interaktif** yang dilengkapi dengan fitur *Text-to-Speech* (TTS) dan logika pewaktuan (timer) untuk memfasilitasi komunikasi dua arah bagi teman-teman disabilitas rungu wicara.

## 👤 Pengembang
* **OkaAlv** ---
* **Afifah Naila** ---
* **Thania Dealva** ---
* **Ali Affrahman** ---
* **Ayman Human Sukma** ---

## ⚙️ Persyaratan Sistem
Pastikan perangkat Anda sudah ter-install:
* Python 3.9 atau lebih baru (Disarankan 3.10)
* Git
* Webcam yang berfungsi dengan baik

---

## 🚀 Panduan Instalasi dan Penggunaan (Cara Mencoba)

Sistem ini sudah dilengkapi dengan model AI (Pre-trained) sehingga Anda **tidak perlu** melakukan *training* ulang. Ikuti langkah-langkah di bawah ini untuk langsung mencobanya:

### 1. Clone Repositori
Buka terminal/Command Prompt, lalu jalankan:
```bash
git clone [https://github.com/OkaAlv/bisindo-realtime-detection.git](https://github.com/OkaAlv/bisindo-realtime-detection.git)
cd bisindo-realtime-detection

2. Buat dan Aktifkan Virtual Environment (Disarankan)
Untuk menghindari konflik library, buat lingkungan virtual khusus:

Windows (PowerShell):

Bash
python -m venv env
Set-ExecutionPolicy Unrestricted -Scope Process  # (Jalankan ini jika terjadi error permission)
.\env\Scripts\activate
Mac/Linux:

Bash
python3 -m venv env
source env/bin/activate
3. Install Dependensi Library
Pastikan virtual environment sudah aktif (terdapat tulisan (env) di terminal), lalu install semua paket yang dibutuhkan:

Bash
pip install -r requirements.txt
4. Unduh Model CNN (Pre-Trained Model)
File model AI (model_bisindo_statis_combined.h5) tidak disertakan di dalam folder utama karena ukurannya yang besar.
👉 Silakan unduh modelnya melalui tab Releases di GitHub ini.
Setelah diunduh, letakkan file .h5 tersebut persis di dalam folder utama proyek ini.

5. Jalankan Aplikasi!
Setelah model tersedia, nyalakan kamera dan jalankan sistem penerjemah utamanya dengan perintah:

Bash
python realtime_tts.py
💡 Cara Penggunaan Aplikasi:

Mengetik Huruf: Bentuk tangan Anda sesuai abjad BISINDO (A-Z) di depan kamera. Tahan posisi tangan selama 1,5 detik agar sistem mengunci dan mencetak huruf tersebut ke layar.

Membaca Kata (Suara): Setelah merangkai satu kata, turunkan tangan Anda dari sorotan kamera selama 2,5 detik. Sistem otomatis membacakan kata tersebut menggunakan suara (Text-to-Speech) dan mengosongkan layar untuk kalimat berikutnya.

Tekan 'q' pada keyboard untuk mematikan sistem.

📂 Penjelasan File Tambahan
Selain program utama, repositori ini juga menyediakan script tambahan bagi yang ingin mempelajari sistem di belakang layar:

train_cnn.py: Script untuk melatih ulang arsitektur CNN jika Anda ingin menambahkan dataset baru.

evaluasi_model.py & evaluasi_validasi.py: Script untuk mencetak Confusion Matrix dan metrik evaluasi model.

testing_file.py: Untuk menguji akurasi model menggunakan file gambar statis atau video tanpa perlu menyalakan webcam.

generate_laporan.py: Menghasilkan visualisasi Feature Maps untuk membedah cara kerja filter CNN dalam mengenali pola tangan.

Proyek ini dikembangkan secara mandiri sebagai bentuk penerapan teknologi Computer Vision untuk menjembatani aksesibilitas komunikasi bahasa isyarat di Indonesia.


***

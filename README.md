# 🤟 Sistem Deteksi Real-time BISINDO (Bahasa Isyarat Indonesia)

Proyek *Computer Vision* ini dibangun menggunakan arsitektur *Pipeline 2-Tahap*: **MediaPipe** untuk ekstraksi *landmark* tangan (lokalisasi) dan **Convolutional Neural Network (CNN)** dengan TensorFlow/Keras untuk klasifikasi bentuk tangan secara *real-time*.

## Pengembang
* OkaAlv


---

## ⚙️ Persyaratan Sistem
Pastikan laptop Anda sudah ter-install:
* Python 3.9 atau lebih baru (Disarankan 3.10)
* Git

---

## 🚀 Panduan Instalasi dan Penggunaan

Ikuti langkah-langkah di bawah ini secara berurutan untuk menjalankan proyek ini di laptop Anda.

### 1. Clone Repositori
Buka terminal/Command Prompt, arahkan ke folder tempat Anda ingin menyimpan proyek ini, lalu jalankan:
```bash
git clone [https://github.com/OkaAlv/bisindo-realtime-detection.git](https://github.com/OkaAlv/bisindo-realtime-detection.git)
cd bisindo-realtime-detection

2. Buat dan Aktifkan Virtual Environment (Sangat Disarankan)
Untuk menghindari konflik antar library (Dependency Hell), buat lingkungan virtual khusus:
Windows:

Bash
python -m venv env
.\env\Scripts\activate
Mac/Linux:

Bash
python3 -m venv env
source env/bin/activate

3. Install Dependensi Library
Pastikan virtual environment sudah aktif (ada tulisan (env) di terminal), lalu install semua paket yang dibutuhkan:

Bash
pip install -r requirements.txt
4. Latih Model CNN (WAJIB!)
PENTING: File model_bisindo_statis.h5 tidak disertakan di repositori ini karena melebihi batas ukuran GitHub (100MB). Anda harus melatih modelnya sendiri agar laptop Anda menghasilkan file .h5 tersebut.

Bash
python train_cnn.py
Tunggu hingga proses Epoch 1/10 sampai 10/10 selesai dan terminal menampilkan pesan bahwa model berhasil disimpan.

5. Jalankan Deteksi Real-time
Setelah file model_bisindo_statis.h5 berhasil terbuat, Anda bisa langsung menyalakan kamera dan menguji deteksinya:

Bash
python deteksi_realtime.py
Cara Penggunaan: Bentuk tangan sesuai abjad BISINDO di depan kamera. Tahan posisi tangan selama sekitar setengah detik agar sistem merangkainya menjadi sebuah kata di bagian bawah layar.
Tekan tombol C pada keyboard untuk menghapus teks, dan Q untuk mematikan kamera.

📂 Penjelasan File Lainnya
Selain program utama, repositori ini juga dilengkapi dengan script untuk keperluan testing dan laporan (visualisasi):

testing_file.py : Digunakan untuk menguji akurasi model menggunakan file gambar (.jpg/.png) atau video rekaman (.mp4) tanpa perlu menyalakan kamera real-time.

generate_laporan.py : Menghasilkan visualisasi input RGB dan Feature Maps (lapisan Conv2D pertama) untuk melihat cara kerja filter CNN.

visualisasi_modul_lengkap.py : Menghasilkan visualisasi lengkap dari tahap augmentasi, normalisasi, hingga Max Pooling.

visualisasi_semua_tahapan.py : Script Sapu Jagat yang membedah layer demi layer dari model .h5 dan menyimpannya ke dalam folder laporan_tahapan_cnn (Sangat cocok untuk presentasi/sidang).

Proyek ini dikembangkan untuk memenuhi tugas mata kuliah Computer Vision tingkat S1 (Semester 6).

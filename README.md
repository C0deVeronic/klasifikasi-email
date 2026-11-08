# 📧 Analisis & Klasifikasi Email Spam (Tugas Statpro)

Proyek ini dibuat untuk mata kuliah **Statistika dan Probabilitas (Statpro)** dengan topik implementasi **Teorema Bayes (Multinomial Naïve Bayes)** pada dataset klasifikasi pesan spam.

---

## 📁 Struktur File
```
Statpro/
├── Dataset/
│   └── spam.csv                             # Dataset SMS / Email Spam
├── Klasifikasi_Spam_Email_Statpro.ipynb      # Notebook Google Colab lengkap
├── app.py                                   # Aplikasi Web Interaktif Streamlit
├── requirements.txt                         # Daftar library Python
└── README.md                                # Panduan proyek
```

---

## 🚀 Cara Menjalankan

### Opsi 1: Google Colab (Notebook)
1. Buka browser dan kunjungi [Google Colab](https://colab.research.google.com).
2. Klik tab **Upload** dan unggah file `Klasifikasi_Spam_Email_Statpro.ipynb`.
3. Unggah file `spam.csv` (dari folder `Dataset`) ke panel Files Colab di sebelah kiri.
4. Jalankan seluruh sel kode secara berurutan (*Runtime -> Run all*).

---

### Opsi 2: Aplikasi Website Lokal (Streamlit)
1. Buka Terminal / PowerShell di folder ini:
   ```bash
   cd "D:\Tugas Kuliah\Statpro"
   ```
2. Instal library yang dibutuhkan:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
4. Browser akan otomatis terbuka di alamat: `http://localhost:8501`.

---

### Opsi 3: Deploy Online Gratis (Agar Bisa Dibuka Siapa Saja)
1. Unggah folder proyek ini ke repositori publik di **GitHub**.
2. Masuk ke [share.streamlit.io](https://share.streamlit.io) menggunakan akun GitHub Anda.
3. Klik **"New app"**, lalu pilih:
   - **Repository:** Pilih repositori GitHub Anda.
   - **Main file path:** `app.py`
4. Klik **Deploy!** Dalam 1-2 menit, Anda akan mendapatkan URL web publik (contoh: `https://spam-email-statpro.streamlit.app`) yang bisa dikirimkan ke dosen atau asisten laboratorium.


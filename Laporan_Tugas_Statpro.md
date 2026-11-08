# LAPORAN TUGAS BESAR STATISTIKA DAN PROBABILITAS
## KLASIFIKASI PESAN SPAM MENGGUNAKAN TEOREMA BAYES (MULTINOMIAL NAÏVE BAYES)

---

### **Disusun Untuk Memenuhi Tugas Mata Kuliah:**
**Statistika dan Probabilitas (Statpro)**

---

## BAB I: PENDAHULUAN

### 1.1 Latar Belakang
Di era komunikasi digital, email dan pesan instan (SMS) telah menjadi media komunikasi primer. Namun, seiring dengan peningkatan volume lalu lintas komunikasi elektronik, ancaman pesan sampah (*spam*)—termasuk penipuan finansial, promosi ilegal, hingga *phishing*—mengalami peningkatan drastis. 

Pesan spam tidak hanya menghabiskan ruang penyimpanan dan bandwidth, namun juga menimbulkan risiko keamanan siber bagi pengguna. Oleh karena itu, diperlukan sistem penyaringan (*filtering*) otomatis yang cerdas, cepat, dan berbasis metode statistik terukur untuk membedakan antara pesan yang sah (**Ham**) dan pesan sampah (**Spam**).

### 1.2 Rumusan Masalah
1. Bagaimana karakteristik sebaran statistik dari pesan Spam dan Ham dalam korpus teks?
2. Bagaimana menerapkan prinsip Teorema Bayes (probabilitas bersyarat) untuk menentukan peluang suatu pesan termasuk kategori Spam berdasarkan kemunculan kata-katanya?
3. Mengapa metrik evaluasi seperti *Precision*, *Recall*, dan *F1-Score* lebih tepat digunakan dibanding *Accuracy* pada dataset yang mengalami ketidakseimbangan kelas (*class imbalance*)?
4. Bagaimana perbandingan performa antara model *Multinomial Naïve Bayes*, *Complement Naïve Bayes*, dan *Logistic Regression*?

### 1.3 Tujuan Proyek
1. Mengembangkan model klasifikasi teks otomatis berbasis Teorema Bayes (*Multinomial Naïve Bayes*).
2. Melakukan analisis statistik deskriptif dan eksplorasi data (*Exploratory Data Analysis*) terhadap distribusi kata dan panjang karakter.
3. Mengimplementasikan aplikasi web interaktif berbasis **Streamlit** dan buku kerja **Google Colab** untuk memfasilitasi pengujian langsung.

---

## BAB II: LANDASAN TEORI

### 2.1 Teorema Bayes (Bayes' Theorem)
Teorema Bayes adalah teorema fundamental dalam teori peluang yang menyatakan hubungan antara probabilitas bersyarat suatu kejadian berdasarkan pengetahuan awal kondisi terkait:

$$P(C \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid C) \cdot P(C)}{P(\mathbf{x})}$$

Di mana:
* $C \in \{\text{Spam}, \text{Ham}\}$: Kelas target.
* $\mathbf{x} = (x_1, x_2, \dots, x_n)$: Vektor fitur kata yang terkandung dalam pesan.
* $P(C \mid \mathbf{x})$: **Posterior Probability** (Peluang pesan berkategori $C$ setelah kata-kata $\mathbf{x}$ diamati).
* $P(C)$: **Prior Probability** (Proporsi frekuensi awal kelas $C$ pada populasi data latih).
* $P(\mathbf{x} \mid C)$: **Likelihood** (Peluang sekumpulan kata $\mathbf{x}$ muncul jika diketahui pesan tersebut berkelas $C$).
* $P(\mathbf{x})$: **Evidence / Marginal Probability** (Faktor normalisasi total kemunculan kata).

### 2.2 Asumsi Naïve (Conditional Independence)
Untuk mengatasi masalah dimensi fitur yang sangat besar dalam teks (*curse of dimensionality*), algoritma Naïve Bayes mengasumsikan bahwa seluruh fitur kata $x_i$ saling bebas secara kondisional jika diberikan kelas $C$:

$$P(\mathbf{x} \mid C) = \prod_{i=1}^n P(x_i \mid C)$$

Sehingga penentuan kelas akhir dilakukan dengan memilih argumen probabilitas maksimum (*Maximum A Posteriori / MAP*):

$$\hat{y} = \arg\max_{C} \left( \log P(C) + \sum_{i=1}^n \log P(x_i \mid C) \right)$$

### 2.3 Laplace Smoothing
Apabila sebuah kata $x_i$ pada saat pengujian belum pernah muncul pada data latih kelas $C$, maka nilai $P(x_i \mid C) = 0$. Perkalian probabilitas nol ini akan menyebabkan seluruh nilai posterior menjadi nol (*Zero Probability Problem*). Untuk mengatasinya, digunakan teknik *Laplace Smoothing* ($\alpha = 1.0$):

$$\hat{P}(x_i \mid C) = \frac{N_{x_i, C} + \alpha}{N_C + \alpha \cdot |V|}$$

Di mana $N_{x_i, C}$ adalah total frekuensi kata $x_i$ pada kelas $C$, $N_C$ adalah total kata pada kelas $C$, dan $|V|$ adalah ukuran perbendaharaan kata (*vocabulary size*).

### 2.4 Pembobotan Fitur: TF-IDF
Teks dikonversi menjadi representasi numerik menggunakan bobot **Term Frequency - Inverse Document Frequency (TF-IDF)**:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left( \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} \right) + 1$$

Fungsi ini memberikan bobot tinggi pada kata-kata khas pembeda dan meredam kata-kata umum yang sering muncul di semua dokumen.

### 2.5 Metrik Evaluasi Data Tidak Seimbang (Class Imbalance)
Pada klasifikasi spam:
* **True Positive (TP):** Pesan spam terdeteksi sebagai spam.
* **True Negative (TN):** Pesan ham terdeteksi sebagai ham.
* **False Positive (FP):** Pesan ham (penting) keliru dibuang ke spam (Sangat Merugikan Pengguna).
* **False Negative (FN):** Pesan spam lolos ke kotak masuk.

Rumus metrik:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
$$\text{Precision} = \frac{TP}{TP + FP} \quad (\text{Fokus utama: Meminimalkan False Positive})$$
$$\text{Recall} = \frac{TP}{TP + FN} \quad (\text{Sensitivitas deteksi spam})$$
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## BAB III: METODOLOGI PENELITIAN

1. **Pengumpulan & Validasi Data:**
   * Dataset memuat ~5.572 pesan SMS/Email berbahasa Inggris.
   * Pemeriksaan *missing value* dan pembersihan baris duplikat.
2. **Text Preprocessing:**
   * *Case folding* (mengubah teks menjadi huruf kecil).
   * Pembersihan URL, tanda baca, simbol non-ASCII, dan angka via ekspresi reguler (*regex*).
   * *Stopwords removal* menggunakan pustaka NLTK bahasa Inggris.
   * *Stemming* menggunakan algoritma *Porter Stemmer*.
3. **Ekstraksi Fitur:**
   * Pembobotan TF-IDF dengan pembatasan 2.500 fitur paling informatif dan rentang n-gram (1, 2) untuk menangkap konteks gabungan dua kata.
4. **Stratified Train-Test Split:**
   * Pembagian 80% data latih dan 20% data uji dengan parameter stratifikasi agar proporsi spam pada kedua subset data tetap identik dengan populasi aslinya (~86,6% vs ~13,4%).
5. **Pelatihan Tiga Model:**
   * **Multinomial Naïve Bayes**
   * **Complement Naïve Bayes**
   * **Logistic Regression**

---

## BAB IV: HASIL DAN ANALISIS

### 4.1 Analisis Statistik Deskriptif (EDA)
* **Distribusi Kelas:** Ham mendominasi sebesar 86,6%, sedangkan Spam sebesar 13,4%.
* **Panjang Karakter:** 
  * Pesan **Ham** memiliki rata-rata panjang karakter pendek (~70 karakter) dengan sebaran luas.
  * Pesan **Spam** memiliki rata-rata panjang karakter yang jauh lebih tinggi (~140-160 karakter) karena pembuat spam memaksimalkan kuota pesan untuk memuat nomor kontak, tautan, dan iming-iming hadiah.

### 4.2 Analisis Rasio Peluang Log (Log-Odds Ratio)
Dari estimasi probabilitas Multinomial Naïve Bayes, kata-kata yang memiliki nilai $\log P(w \mid \text{Spam}) - \log P(w \mid \text{Ham})$ tertinggi adalah:
1. *claim*
2. *prize*
3. *urgent*
4. *won*
5. *call*

Kata-kata ini secara statistik memiliki signifikansi paling tinggi sebagai penanda konten spam.

### 4.3 Performa Evaluasi Model
Model **Multinomial Naïve Bayes** menghasilkan tingkat **Precision mendekati 100% (hampir 0 False Positive)**. Hal ini sangat ideal untuk filter email di dunia nyata karena mencegah pesan penting pengguna keliru disaring ke kotak spam.

---

## BAB V: KESIMPULAN

1. Teorema Bayes melalui model **Multinomial Naïve Bayes** terbukti sangat efisien, memiliki kecepatan inferensi tinggi, dan menghasilkan tingkat presisi optimal dalam menyaring pesan spam.
2. Pembobotan **TF-IDF** berhasil menyeleksi fitur linguistik diskriminatif antara pesan sah dan pesan penipuan.
3. Aplikasi web **Streamlit** dan buku kerja **Google Colab** yang dibangun memberikan kemudahan bagi pengujian interaktif baik secara individual maupun pemrosesan massal (*batch processing*).


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Klasifikasi Spam Email - Statpro",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling sederhana & bersih
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Unduh resource NLTK dengan aman
@st.cache_resource
def setup_nltk():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    except Exception:
        pass

setup_nltk()

ps = PorterStemmer()
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set(["i", "me", "my", "myself", "we", "our", "you", "your", "he", "him", "she", "it", "they", "what", "which", "who", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    cleaned = [ps.stem(w) for w in words if w not in stop_words and len(w) > 1]
    return ' '.join(cleaned)

# ==========================================
# CACHE DATA & TRAINING MODEL
# ==========================================
@st.cache_resource(show_spinner="Sedang memuat data dan melatih model probabilitas...")
def load_data_and_train():
    # Menemukan path dataset
    possible_paths = [
        'Dataset/spam.csv',
        'spam.csv',
        os.path.join(os.path.dirname(__file__), 'Dataset', 'spam.csv'),
        os.path.join(os.path.dirname(__file__), 'spam.csv')
    ]
    file_path = None
    for p in possible_paths:
        if os.path.exists(p):
            file_path = p
            break
            
    if file_path is None:
        return None, None, None, None, None, None

    # Load file
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin-1')

    if 'Unnamed: 0' in df.columns or df.columns[0] == '':
        df = df.iloc[:, 1:]
    
    df.columns = ['label', 'message']
    df = df.drop_duplicates(keep='first').reset_index(drop=True)
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Statistik panjang teks
    df['char_length'] = df['message'].apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    
    # Preprocessing
    df['clean_message'] = df['message'].apply(clean_text)
    
    # TF-IDF
    tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
    X = tfidf.fit_transform(df['clean_message']).toarray()
    y = df['label_num'].values
    
    # Stratified Train-Test Split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Training Model
    models = {
        "Multinomial Naïve Bayes": MultinomialNB(alpha=1.0),
        "Complement Naïve Bayes": ComplementNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }
    
    trained_models = {}
    eval_metrics = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        eval_metrics[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': auc(fpr, tpr),
            'cm': confusion_matrix(y_test, y_pred),
            'fpr': fpr,
            'tpr': tpr
        }
        
    return df, tfidf, trained_models, eval_metrics, X_test, y_test

# Load data dan model
df, tfidf, models, metrics, X_test, y_test = load_data_and_train()

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2875/2875394.png", width=70)
    st.title("Email Spam Filter")
    st.caption("Aplikasi Klasifikasi Berbasis Teorema Bayes")
    st.markdown("---")
    
    st.subheader("⚙️ Pengaturan Model")
    if models is not None:
        selected_model_name = st.selectbox(
            "Pilih Algoritma Klasifikasi:",
            list(models.keys()),
            index=0,
            help="Multinomial Naïve Bayes adalah model utama berbasis Teorema Bayes."
        )
    else:
        selected_model_name = "Model tidak tersedia"
        
    st.markdown("---")
    st.markdown("""
    **Mata Kuliah:**  
    📚 Statistika dan Probabilitas (Statpro)  
    
    **Topik:**  
    * Teorema Bayes (Bayes' Theorem)
    * Probabilitas Bersyarat & Log-Odds
    * Evaluasi Metrik Data Tidak Seimbang
    """)

# ==========================================
# MAIN CONTENT
# ==========================================
st.markdown('<div class="main-title">📧 Spam Email Classifier & Probability Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Aplikasi web deteksi spam email secara <i>real-time</i> menggunakan Teorema Bayes dan Pembobotan TF-IDF.</div>', unsafe_allow_html=True)

if df is None:
    st.error("⚠️ File dataset `spam.csv` tidak ditemukan di folder `Dataset/` atau folder utama. Silakan pastikan file `spam.csv` telah tersedia.")
    st.stop()

# Buat Tab Navigasi
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Uji Coba & Prediksi Pesan",
    "📊 Analisis Dataset & Statistik (EDA)",
    "📈 Evaluasi & Perbandingan Model",
    "📖 Landasan Teori Probabilitas"
])

# ------------------------------------------
# TAB 1: PREDIKSI INTERAKTIF
# ------------------------------------------
with tab1:
    st.subheader("Tes Pesan Email / SMS")
    st.write("Masukkan teks pesan di bawah ini untuk melihat apakah sistem mengklasifikasikannya sebagai **Spam** atau **Ham** (bukan spam).")
    
    # Tombol Contoh Cepat
    col_demo1, col_demo2, col_demo3 = st.columns(3)
    sample_text = ""
    if col_demo1.button("📩 Contoh 1: Pesan Hadiah (Spam)"):
        sample_text = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Claim code KL341. Valid 12 hours only. Call 09061701461 now!"
    if col_demo2.button("💬 Contoh 2: Pesan Kuliah (Ham)"):
        sample_text = "Hi mate, don't forget we have a group presentation for the Probability and Statistics class tomorrow morning at 10 AM. See you in class!"
    if col_demo3.button("⚠️ Contoh 3: Promo / Phishing (Spam)"):
        sample_text = "URGENT! Your mobile account balance is credited with £500 bonus. Click http://claim-gift.now to claim before it expires today!"

    user_input = st.text_area(
        "Isi Pesan / Email:",
        value=sample_text,
        height=140,
        placeholder="Ketik atau tempel teks email/pesan di sini..."
    )

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        predict_clicked = st.button("🚀 Prediksi Sekarang", type="primary", use_container_width=True)

    if predict_clicked and user_input.strip() != "":
        # Preprocessing & Prediksi
        cleaned_input = clean_text(user_input)
        input_vec = tfidf.transform([cleaned_input]).toarray()
        
        active_model = models[selected_model_name]
        pred = active_model.predict(input_vec)[0]
        prob = active_model.predict_proba(input_vec)[0]
        
        prob_ham = prob[0] * 100
        prob_spam = prob[1] * 100
        
        st.markdown("---")
        st.subheader("🎯 Hasil Analisis Probabilitas")
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            if pred == 1:
                st.error("### 🚨 TERDETEKSI: SPAM")
                st.markdown(f"Model **{selected_model_name}** mengidentifikasi pesan ini sebagai pesan **SPAM / Penipuan**.")
            else:
                st.success("### ✅ TERDETEKSI: HAM (BUKAN SPAM)")
                st.markdown(f"Model **{selected_model_name}** mengidentifikasi pesan ini sebagai pesan **NORMAL / VALID**.")
                
            st.write(f"**Tingkat Keyakinan (Posterior Probability):**")
            st.progress(int(prob_spam))
            st.markdown(f"- Peluang **Ham** : `{prob_ham:.2f}%`\n- Peluang **Spam**: `{prob_spam:.2f}%`")
            
        with col_res2:
            st.info("💡 **Hasil Pemrosesan Teks (Preprocessed):**")
            st.caption("Teks asli setelah diubah ke huruf kecil, dibersihkan dari simbol/URL/angka, dihapus stop words, dan di-stem:")
            st.code(cleaned_input if cleaned_input != "" else "(Semua kata tereliminasi sebagai stop words)", language="text")
            
            # Kata kunci pembeda yang terdeteksi
            detected_features = [word for word in cleaned_input.split() if word in tfidf.vocabulary_]
            if detected_features:
                st.write(f"Kata kunci terdeteksi dalam model vocab ({len(detected_features)}):")
                st.write(", ".join([f"`{w}`" for w in set(detected_features)]))

# ------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------
with tab2:
    st.subheader("📊 Analisis Karakteristik Dataset (EDA)")
    
    # Metric cards
    total_samples = len(df)
    ham_count = (df['label'] == 'ham').sum()
    spam_count = (df['label'] == 'spam').sum()
    ham_pct = (ham_count / total_samples) * 100
    spam_pct = (spam_count / total_samples) * 100
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Data Bersih", f"{total_samples:,}")
    m2.metric("Pesan Ham (Normal)", f"{ham_count:,}", f"{ham_pct:.1f}%")
    m3.metric("Pesan Spam", f"{spam_count:,}", f"-{spam_pct:.1f}%")
    m4.metric("Rasio Imbalance", f"{ham_count/spam_count:.1f} : 1")
    
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("##### Proporsi Sebaran Kelas")
        fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
        ax_pie.pie([ham_count, spam_count], labels=['Ham', 'Spam'], autopct='%1.1f%%',
                   startangle=90, colors=['#2563EB', '#DC2626'], explode=(0, 0.08),
                   wedgeprops=dict(width=0.45, edgecolor='white'))
        ax_pie.set_title("Proporsi Kelas Pesan", fontweight='bold')
        st.pyplot(fig_pie)
        
    with col_chart2:
        st.markdown("##### Distribusi Panjang Karakter per Pesan")
        fig_len, ax_len = plt.subplots(figsize=(6, 4))
        sns.histplot(df[df['label'] == 'ham']['char_length'], color='#2563EB', label='Ham', kde=True, bins=40, ax=ax_len, alpha=0.5)
        sns.histplot(df[df['label'] == 'spam']['char_length'], color='#DC2626', label='Spam', kde=True, bins=40, ax=ax_len, alpha=0.6)
        ax_len.set_xlim(0, 300)
        ax_len.set_xlabel("Panjang Karakter")
        ax_len.set_ylabel("Frekuensi")
        ax_len.legend()
        st.pyplot(fig_len)
        
    st.info("""
    **Insight Statistika:**  
    - Dataset mengalami **Class Imbalance** yang signifikan (~87% ham vs ~13% spam).
    - Pesan **Spam** memiliki rata-rata panjang karakter yang jauh lebih tinggi dan konsisten (~140-160 karakter), sedangkan pesan **Ham** cenderung jauh lebih pendek dan variatif.
    """)
    
    with st.expander("🔍 Pratinjau Sampel Data Mentah"):
        st.dataframe(df[['label', 'message', 'char_length', 'word_count']].head(15), use_container_width=True)

# ------------------------------------------
# TAB 3: EVALUASI & PERBANDINGAN MODEL
# ------------------------------------------
with tab3:
    st.subheader("📈 Evaluasi Kinerja Model Statistik")
    st.write("Perbandingan performa ketiga model pada data uji (*Testing Set* 20%) yang telah distratifikasi.")
    
    # Ringkasan tabel
    summary_data = []
    for m_name, val in metrics.items():
        summary_data.append({
            "Nama Model": m_name,
            "Accuracy": f"{val['Accuracy']*100:.2f}%",
            "Precision (Spam)": f"{val['Precision']*100:.2f}%",
            "Recall (Spam)": f"{val['Recall']*100:.2f}%",
            "F1-Score (Spam)": f"{val['F1-Score']*100:.2f}%",
            "ROC-AUC": f"{val['ROC-AUC']:.4f}"
        })
    df_metrics = pd.DataFrame(summary_data)
    st.dataframe(df_metrics, use_container_width=True)
    
    st.markdown("---")
    
    col_cm, col_roc = st.columns(2)
    
    with col_cm:
        st.markdown(f"##### Confusion Matrix: {selected_model_name}")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        cm = metrics[selected_model_name]['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['Pred Ham', 'Pred Spam'],
                    yticklabels=['Aktual Ham', 'Aktual Spam'])
        ax_cm.set_ylabel("Label Sebenarnya")
        ax_cm.set_xlabel("Label Hasil Prediksi")
        st.pyplot(fig_cm)
        
    with col_roc:
        st.markdown("##### Kurva ROC (Receiver Operating Characteristic)")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4.3))
        for m_name, val in metrics.items():
            ax_roc.plot(val['fpr'], val['tpr'], lw=2, label=f"{m_name} (AUC={val['ROC-AUC']:.3f})")
        ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax_roc.set_xlabel("False Positive Rate (FPR)")
        ax_roc.set_ylabel("True Positive Rate (Recall)")
        ax_roc.legend(loc="lower right", fontsize=8)
        st.pyplot(fig_roc)

    st.markdown("""
    > **Catatan Analisis:**  
    > Pada sistem filter spam, **Precision** tinggi sangat krusial untuk mencegah terjadinya *False Positive* (pesan penting keliru dibuang ke folder spam). Model **Multinomial Naïve Bayes** menunjukkan tingkat *Precision* yang luar biasa tinggi pada kasus ini.
    """)

# ------------------------------------------
# TAB 4: LANDASAN TEORI BAYES
# ------------------------------------------
with tab4:
    st.subheader("📖 Teorema Bayes dalam Klasifikasi Teks")
    st.markdown(r"""
    ### 1. Rumus Dasar Teorema Bayes
    Teorema Bayes menyatakan hubungan antara probabilitas bersyarat dari dua kejadian acak:
    
    $$P(C \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid C) \cdot P(C)}{P(\mathbf{x})}$$
    
    Dalam konteks klasifikasi email spam:
    * $P(C \mid \mathbf{x})$: **Posterior Probability**, yaitu peluang sebuah email masuk ke kelas $C \in \{\text{Spam}, \text{Ham}\}$ setelah membaca sekumpulan kata $\mathbf{x} = (x_1, x_2, \dots, x_n)$.
    * $P(C)$: **Prior Probability**, yaitu frekuensi proporsi kemunculan masing-masing kelas dalam dataset.
    * $P(\mathbf{x} \mid C)$: **Likelihood**, yaitu peluang sekumpulan kata $\mathbf{x}$ muncul jika pesan tersebut adalah kelas $C$.
    * $P(\mathbf{x})$: **Evidence**, faktor normalisasi probabilitas konstan.

    ---
    ### 2. Asumsi Naïve (Conditional Independence)
    Kata *"Naïve"* berasal dari asumsi bahwa setiap fitur kata $x_i$ saling bebas (*independen*) terhadap kata lainnya jika diberikan kelas $C$:
    
    $$P(\mathbf{x} \mid C) = \prod_{i=1}^n P(x_i \mid C)$$
    
    ### 3. Laplace Smoothing
    Untuk menghindari nilai probabilitas bernilai $0$ pada kata baru yang belum pernah muncul pada kelas tertentu saat *training*, diterapkan **Laplace Smoothing** ($\alpha = 1$):
    
    $$\hat{P}(x_i \mid C) = \frac{N_{x_i, C} + \alpha}{N_C + \alpha \cdot |V|}$$
    """)

st.markdown("---")
st.caption("Aplikasi Dibuat untuk Tugas Kuliah Statistika dan Probabilitas (Statpro) • Powered by Streamlit & Scikit-Learn")


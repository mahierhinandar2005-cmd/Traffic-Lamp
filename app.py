import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# =====================================================================
# 1. LOAD MODEL AI YANG SUDAH DITRAINING
# =====================================================================
@st.cache_resource
def load_my_model():
    # Membaca file model .h5 hasil training dari train.py
    return tf.keras.models.load_model('model_lampu_lalin.h5')

try:
    model = load_my_model()
    # Sesuaikan susunan kelas dengan folder: [green, red, yellow]
    class_names = ['Hijau (Green)', 'Merah (Red)', 'Kuning (Yellow)']
except:
    st.error("❌ File 'model_lampu_lalin.h5' tidak ditemukan! Jalankan train.py dulu sampai selesai ya.")

# =====================================================================
# 2. CONFIGURASI INTERFACE WEB STREAMLIT
# =====================================================================
st.set_page_config(page_title="Deteksi Lampu Lalu Lintas", page_icon="🚦")

st.title("🚦 Aplikasi Deteksi Warna Lampu Lalu Lintas")
st.write("Proyek AI Kelompok 5 - Unggah foto lampu lalu lintas untuk dideteksi oleh AI.")

# Komponen UI untuk Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar lampu lalin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Tampilkan gambar yang diunggah user di web
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang Diunggah', use_column_width=True)
    
    st.write("🔄 Sedang menganalisis gambar...")
    
    # 2. Preprocessing Gambar (Samakan dengan settingan train.py versi 50 Epochs)
    # Ukuran diubah ke 64x64 piksel sesuai dengan model train terbaru kamu
    img_resized = image.resize((64, 64)) 
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0) # Menambah dimensi batch
    
    # 3. Prediksi Menggunakan Model AI
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    # Mengambil index kelas dengan nilai kecocokan tertinggi
    hasil_prediksi = class_names[np.argmax(score)]
    tingkat_akurasi = 100 * np.max(score)
    
    # =====================================================================
    # 3. LOGIKA OUTPUT WARNA LAMPU (SINKRON DENGAN KETENTUAN TUGAS)
    # =====================================================================
    st.success("🎉 Analisis Selesai!")
    
    # Menampilkan hasil deteksi utama
    st.subheader(f"Hasil Deteksi: **{hasil_prediksi}**")
    st.write(f"Tingkat Keyakinan AI: **{tingkat_akurasi:.2f}%**")
    
    st.markdown("---")
    st.write("### 📌 Status Tindakan Pengendara:")
    
    # Logika if-else untuk menampilkan perintah berkendara berdasarkan warna
    if "Merah" in hasil_prediksi:
        st.error("🚨 **KENDARAAN WAJIB BERHENTI!** Jangan menerobos garis marka jalan.")
    elif "Kuning" in hasil_prediksi:
        st.warning("⚠️ **PERINGATAN!** Kendaraan harus mengurangi kecepatan dan bersiap untuk berhenti.")
    elif "Hijau" in hasil_prediksi:
        st.info("✅ **KENDARAAN DIPERBALEHKAN JALAN!** Tetap utamakan keselamatan dan perhatikan sekitar.")
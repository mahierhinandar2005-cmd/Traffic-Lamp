import tensorflow as tf
from tensorflow.keras import layers, models
import os

# =====================================================================
# 1. KONFIGURASI PARAMETER (VERSI 50 EPOCHS)
# =====================================================================
BATCH_SIZE = 32
# Kita perkecil ke 64x64 agar gambar lebih buram, AI jadi butuh epoch banyak untuk belajar
IMG_SIZE = (64, 64) 
EPOCHS = 50 # Sesuai request, biar sama kayak temen-temen!

TRAIN_DIR = "train"
VAL_DIR = "val"

if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
    print("❌ ERROR: Folder 'train' atau 'val' tidak ditemukan!")
    exit()

# =====================================================================
# 2. MEMUAT DATASET GAMBAR
# =====================================================================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
class_names = train_ds.class_names

# =====================================================================
# 3. ARSITEKTUR MODEL (DIBIKIN AGAK SULIT MENGHAFAL)
# =====================================================================
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(64, 64, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    
    # Pasang dua layer Dropout biar AI-nya gak gampang dapet 96%
    layers.Dropout(0.6), 
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5), 
    
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =====================================================================
# 4. PROSES TRAINING
# =====================================================================
print(f"\n🚀 Memulai training AI selama {EPOCHS} Epochs...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# =====================================================================
# 5. SIMPAN MODEL
# =====================================================================
model.save('model_lampu_lalin.h5')
print("\n✅ SELESAI! Model 50 Epochs berhasil disimpan.")
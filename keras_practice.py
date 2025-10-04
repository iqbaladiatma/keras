import keras
import numpy as np

# ================================================================
# CONTOH CARA MEMBUAT MODEL (COMMENTED EXAMPLES)
# ================================================================

# Cara #1: Pakai list langsung
# model = keras.Sequential([
#   keras.layers.Flatten(input_shape=(28, 28)),
#   keras.layers.Dense(128, activation='relu'),
#   keras.layers.Dense(10, activation='softmax'),
# ])

# Cara #2: Membuat wadah kosong, lalu menambahkan layer satu per satu
# model = keras.Sequential()
# model.add(keras.Input(shape=(28,28))) # Mendefinisikan bentuk input
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dense(10, activation='softmax'))

# --- 1. MEMBUAT DATA TIRUAN ---
# Kita buat 100 "gambar" acak berukuran 28x28 piksel
# dan 100 "label" acak untuk 10 kelas.
jumlah_data = 100
x_train = np.random.rand(jumlah_data, 28, 28)
# Label harus dalam format one-hot encoding karena kita pakai 'categorical_crossentropy'
y_train = keras.utils.to_categorical(np.random.randint(10, size=jumlah_data))

# --- 2. MEMBANGUN MODEL (CARA #2) ---
print("Membangun arsitektur model...")
model = keras.Sequential()
model.add(keras.Input(shape=(28, 28)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

# --- 3. MENGOMPILASI MODEL ---
print("Mengompilasi model...")
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=['accuracy'])
# optimizer: Bagaimana model akan belajar dan memperbaiki diri.
# loss: Bagaimana cara mengukur kesalahan model.
# metrics: Apa "nilai rapor" yang ingin kamu lihat.


# --- 4. MENAMPILKAN STRUKTUR MODEL ---
print("\n--- Struktur Model (Output dari model.summary()) ---")
model.summary()

# --- 5. MELATIH MODEL DENGAN DATA TIRUAN ---
print("\n--- Proses Pelatihan (Output dari model.fit()) ---")
model.fit(x_train, y_train, epochs=3, batch_size=16)




# ================================================================
# CONTOH KEDUA: VERSI TERPISAH (BLUEPRINT + TRAINING)
# ================================================================
# Ini contoh cara pisah antara desain model dan training
# Di Keras 3, cara import utamanya adalah langsung dari 'keras'

# ================================================================
# BAGIAN 1: DESAIN ARSITEKTUR (BLUEPRINT)
# ================================================================

print("Tahap 1: Merancang Blueprint Model dengan Keras 3...")

# Membuat model Sequential. Caranya tetap sama.
model = keras.Sequential()
# Menentukan bentuk input dan menambahkan lapisan (layers)
# Caranya juga tidak berubah.
model.add(keras.Input(shape=(784,))) # Input: 784 angka
model.add(keras.layers.Dense(128, activation='relu')) # Hidden Layer
model.add(keras.layers.Dense(10, activation='softmax')) # Output Layer

# ================================================================
# BAGIAN 2: KOMPILASI MODEL
# ================================================================

print("\nTahap 2: Mengonfigurasi Model...")

# Mengompilasi model. Caranya juga masih sama.
# Optimizer, loss, dan metrics bisa dipanggil dari keras.optimizers, dll.
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

# Menampilkan blueprint
model.summary()


# ================================================================
# BAGIAN 3: MENYAMBUNGKAN DENGAN DATA & MELATIH
# ================================================================

print("\nTahap 3: Menyiapkan Data & Memulai Training...")

# Membuat data tiruan (dummy data)
x_train = np.random.rand(500, 784) # 500 contoh data
y_train = np.random.randint(10, size=500) # 500 label (angka 0-9)

# Melatih model. Perintahnya tetap sama: model.fit()
model.fit(x_train, y_train, epochs=5, batch_size=32)

print("\nTraining Selesai!")
# =================================================
# LATIHAN PERTAMA KERAS 3
# Klasifikasi Gambar Fashion-MNIST
# =================================================

# 1. Impor library yang dibutuhkan
print("Mengimpor library...")
import keras
import numpy as np

# 2. Memuat dataset
print("Memuat dataset Fashion-MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# 3. Memproses data (Normalisasi)
print("Memproses data...")
x_train = x_train / 255.0
x_test = x_test / 255.0

# 4. Membangun arsitektur model
print("Membangun model...")
model = keras.Sequential([
    keras.Input(shape=(28, 28)),          # Input gambar 28x28 piksel
    keras.layers.Flatten(),               # Meratakan gambar menjadi 1D
    keras.layers.Dense(128, activation='relu'), # Lapisan tersembunyi
    keras.layers.Dense(10, activation='softmax')  # Lapisan output untuk 10 kelas
])

# 5. Mengompilasi model
print("Mengompilasi model...")
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Melatih model
print("Memulai pelatihan model...")
model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)

# 7. Mengevaluasi model
print("\nMengevaluasi performa model...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nAkurasi pada data uji: {test_acc*100:.2f}%")

print("\n===== Selesai! =====")
import keras
import numpy as np

model = keras.Sequential()
# membuat objek model yang kosong

# menentukan pintu masuk
model.add(keras.Input(shape=(784,)))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

# konfigurasi cara belajar
model.compile(
  # paramater 1 : Si Mekanik, nyewa nih
  optimizer=keras.optimizers.Adam(learning_rate=0.001),

  # Parameter 2: Si Penghitung Error
  loss=keras.losses.SparseCategoricalCrossentropy(),

  # Parameter 3: Si Pelapor Performa
  metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

# menampilkan dalam bentuk tabel, arsitekturnya
model.summary()

# menyiapkan data untuk dilatih
x_train = np.random.rand(500, 784)
# kita membuat 500 contoh data dimana setiap contohnya isinya 784 angka acak
y_train = np.random.randint(10, size=500)
# kita membuat 500 jawaban acak yang berisi angka dari 0 hingga 9

# melatih model
model.fit(x_train, y_train, epochs=5, batch_size=32)
# perintah mulai belajar
# model.fit(...): Inilah perintah utamanya: "Mulai Belajar!".
# x_train, y_train: Kamu memberikan "soal" dan "kunci jawaban" ke model.
# epochs=5: Kamu menyuruh model untuk mengulangi proses belajar dari keseluruhan data sebanyak 5 kali putaran.
# batch_size=32: Daripada belajar dari 500 data sekaligus, model akan belajar dari 32 data dulu, lalu mengoreksi diri, lalu lanjut ke 32 data berikutnya, dan seterusnya. Ini membuat proses belajar lebih efisien.


# # Lakukan ujian!
# CONTOH ATAS ITU TANPA EVALUASI

# Contoh dengan evaluasi
import keras
import numpy as np
from sklearn.model_selection import train_test_split

# membuat semua data mentah
print("Tahap 1 : Menyiapkan Seluruh Data...")
# membuat 1000 contoh data, dengan 784 masing2
semua_soal = np.random.rand(1000, 784)

# membuat kunci jawaban, untuk bank soal diatas
semua_jawaban = np.random.randint(10, size=1000)

print(f"Total data yang dimiliki: {len(semua_soal)} contoh")

# tahap 2 membagi data menjadi set latihan dan ujian
print("\nTahap 2: Membagi data...")

# Menggunakan train_test_split untuk membagi data
# test_size=0.2 artinya kita ambil 20% dari total data untuk ujian

x_train, x_test, y_train, y_test = train_test_split(
  semua_soal,
  semua_jawaban,
  test_size=0.2,
  random_state=42
)

print(f"Jumlah data latihan (train): {len(x_train)}")
print(f"Jumlah data ujian (test): {len(x_test)}")

# membuat,mengcompilasi, melatih model
print("\nTahap 3: Membangun dan melatih model...")

# ini pakai yang list
model = keras.Sequential([
    keras.Input(shape=(784, )),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# compile
model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)
# latih model hanya dengan train
model.fit(x_train, y_train, epochs=5, verbose=1) # verbose=1 untuk menampilkan proses

# menguji model dengan data ujian
print("\nTahap 4: Menguji model dengan data baru...")

# uji model dengan test
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f"\nHasil Ujian Akhir:")
print(f"  - Skor Loss di data ujian: {loss:.4f}")
print(f"  - Skor Akurasi di data ujian: {accuracy:.4f} ({accuracy:.2%})")
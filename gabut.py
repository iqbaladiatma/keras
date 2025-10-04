import keras
import numpy as np

# membuat objek seq kosong, bisa dengan array atau add. 
model = keras.Sequential([
  keras.Input(shape=(784, )),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(10, activation='softmax')
])
# konfigurasi cara belajar/compile
model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)
# menampilkan tabel dengan summary
model.summary()
# siapin data latih x,y
x_train = np.random.rand(1000, 784)
y_train = np.random.randint(10, size=1000)
# latih model dengan fit,epochs,batch_size=32/verbose untuk log tidak terlihat
history = model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

# menampilkan akurasi final dari history training
final_accur = history.history['accuracy'][-1]
print(f"\nAkurasi Akhir pada Data Latih: {final_accur:.4f}")

# model predict = menggunakan model diatas untuk menebak data baru.
# jadi mengerjakan tugas akhir kan ta
# membuat data baru
x_test = np.random.rand(5, 784)

# prediksi dengan model.predict()
predictions = model.predict(x_test)

# print("\nHasil Prediksi (Probabilitas untuk setiap kelas):")
# print(predictions)

# sampai atas cukup tapi terjemahkan info ini ke bahasa yang gampang

# ambil kelas dengan prob palingtingii
pre_class = np.argmax(predictions, axis=1)
# ambil nilai prob kredensial dari kls predict
con_sc = np.max(predictions, axis=1)

# tampilkan data dengan loop
for i in range(len(x_test)):
  predik = pre_class[i]
  kep = con_sc[i]

  # format string
  print(f"Data ke-{i+1}:")
  print(f" => Prediksi Model: Kelas {predik}")
  print(f" => Tingkat kepercayaan: {kep * 100:.2f} %\n")
  # itu : untuk memisahkan . agar mudah dibaca
  # print(f"Hasilnya adalah: {kep * 100} %")
# Output: Hasilnya adalah: 87.654321 % 
# print(f"Hasilnya adalah: {kep * 100:.2f} %")
# Output: Hasilnya adalah: 87.65 %
  # Dikalikan 100 jadi persen, .2f untuk 2 angka di belakang koma

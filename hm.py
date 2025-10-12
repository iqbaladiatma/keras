import keras
import numpy as np
print("lIBRARY BERHASIL DIPANGIL LANGKAH SELANJUTNYA")

model = keras.Sequential([
    keras.Input(shape=(784 ,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

x_train = np.random.rand(1000, 784)
y_train = np.random.randint(10, size=1000)
history = model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
final_accur = history.history['accuracy'][-1]
print(f"\nAkurasi Akhir pada Data Latih: {final_accur:.4f}")
x_test = np.random.rand(5, 784)
predictions = model.predict(x_test)
pre_class = np.argmax(predictions, axis=1)
con_sc = np.max(predictions, axis=1)

for i in range(len(x_test)):
  predik = pre_class[i]
  kep = con_sc[i]

  print(f"Data ke-{i+1}:")
  print(f" => Prediksi Model: Kelas {predik}")
  print(f" => Tingkat kepercayaan: {kep * 100:.2f} %\n")

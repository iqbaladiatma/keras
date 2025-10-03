import numpy as np
import pandas as pd 
import keras

# 1. MEMUAT DATA
# Kita gunakan pandas untuk membaca file CSV
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print("Contoh data latih:")
print(train_df.head())
print("\nJumlah data latih:", len(train_df))
print("Jumlah data uji:", len(test_df))

# Memisahkan teks dan label dari data latih
train_texts = train_df['text'].values
train_labels = train_df['target'].values

# 2. VEKTORISASI TEKS
VOCAB_SIZE = 10000  # Kita batasi jumlah kata unik yang paling sering muncul
SEQUENCE_LENGTH = 100 # Kita batasi panjang setiap tweet menjadi 100 token

# Membuat layer TextVectorization
vectorize_layer = keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)

# Pelajari kosakata dari data teks latih kita
vectorize_layer.adapt(train_texts)

# Ubah teks latih kita menjadi urutan angka (vektor)
x_train = vectorize_layer(train_texts)

print("\nContoh teks asli:", train_texts[0])
print("Hasil setelah vektorisasi:", x_train[0])

# 3. MEMBANGUN MODEL
model = keras.Sequential([
    # Input layer tidak perlu didefinisikan secara eksplisit, akan otomatis
    
    # Layer Embedding: Mengubah angka menjadi vektor kata yang bermakna
    keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=128,
        input_length=SEQUENCE_LENGTH
    ),
    
    # Layer LSTM: Memproses urutan kata untuk memahami konteks
    keras.layers.LSTM(64),
    
    # Layer Dense: Lapisan pemikir tambahan
    keras.layers.Dense(64, activation='relu'),
    
    # Layer Output: PENTING!
    # Hanya 1 neuron dengan aktivasi 'sigmoid' untuk klasifikasi biner
    keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# 4. KOMPILASI & PELATIHAN
model.compile(
    loss='binary_crossentropy', # Loss function khusus untuk klasifikasi biner
    optimizer='adam',
    metrics=['accuracy']
)

print("\nMemulai pelatihan...")
model.fit(
    x_train,
    train_labels,
    epochs=5,
    validation_split=0.1 # Sisihkan 10% data latih untuk validasi setiap epoch
)

# 5. MEMBUAT PREDIKSI UNTUK SUBMISSION
print("\nMembuat prediksi pada data uji...")

# Siapkan data uji dengan cara yang sama seperti data latih
test_texts = test_df['text'].values
x_test = vectorize_layer(test_texts)

# Lakukan prediksi
predictions = model.predict(x_test)

# Mengubah probabilitas (misal: 0.78) menjadi 0 atau 1
# Jika probabilitas > 0.5, kita bulatkan menjadi 1 (bencana), jika tidak 0.
submission_labels = (predictions > 0.5).astype(int)

# Membuat file submission.csv
submission_df = pd.DataFrame({'id': test_df['id'], 'target': submission_labels.flatten()})
submission_df.to_csv('submission.csv', index=False)

print("\nFile 'submission.csv' telah dibuat! Siap untuk diunggah ke Kaggle.")

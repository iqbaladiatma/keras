import keras
import numpy as np
import string

# Data kita: beberapa kalimat sederhana tentang AI
data = """
kecerdasan buatan adalah masa depan
deep learning merupakan cabang dari kecerdasan buatan
keras adalah library untuk deep learning
model dilatih menggunakan data
"""

# --- LANGKAH 1: PERSIAPAN & VEKTORISASI TEKS ---

# Pisahkan teks menjadi baris-baris
corpus = data.lower().split("\n")[1:-1] # [1:-1] untuk menghilangkan baris kosong

# Buat layer TextVectorization, ini adalah pengganti Tokenizer
vectorize_layer = keras.layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="int"
)

# Pelajari semua kata unik dari data kita (mirip .fit_on_texts)
vectorize_layer.adapt(corpus)
vocab = vectorize_layer.get_vocabulary()
total_words = len(vocab)

print("Vocabulary:", vocab)
print("Total kata unik:", total_words)


# --- LANGKAH 2: MEMBUAT PASANGAN INPUT (X) DAN OUTPUT (y) ---

input_sequences = []
# Ubah setiap baris menjadi urutan angka
for line in corpus:
    token_list = vectorize_layer([line])[0].numpy()
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Menyamakan panjang semua 'soal' (Padding)
max_sequence_len = max([len(x) for x in input_sequences])
padded_sequences = keras.utils.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Memisahkan 'soal' (X) dan 'jawaban' (y)
X = padded_sequences[:,:-1]
y = padded_sequences[:,-1]

# Mengubah 'jawaban' (y) menjadi one-hot encoding
y = keras.utils.to_categorical(y, num_classes=total_words)


# --- LANGKAH 3: MEMBANGUN MODEL ---

model = keras.Sequential([
    keras.Input(shape=(max_sequence_len-1,)),
    keras.layers.Embedding(total_words, 10),
    keras.layers.LSTM(100),
    keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# --- LANGKAH 4: MELATIH MODEL ---

print("\nMemulai pelatihan...")
model.fit(X, y, epochs=100, verbose=1)


# --- LANGKAH 5: MEMBUAT PREDIKSI ---

seed_text = "kecerdasan buatan"
next_words = 5

print(f"\nKalimat awal: '{seed_text}'")
print("Model akan melanjutkan...")

for _ in range(next_words):
    # Ubah seed_text menjadi urutan angka
    token_list = vectorize_layer([seed_text])[0].numpy()
    # Pad urutan tersebut
    padded_token_list = keras.utils.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    # Lakukan prediksi
    predicted_probs = model.predict(padded_token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs)
    
    # Cari kata yang sesuai dengan index hasil prediksi
    output_word = vocab[predicted_index]
    
    seed_text += " " + output_word

print("Hasil:", seed_text)
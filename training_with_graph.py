import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Dropout
import pickle
import os
import re
import string

# Metin temizleme fonksiyonu
def clean_text(text):
    """
    Metinden noktalama işaretlerini temizler ve küçük harfe dönüştürür
    """
    # Küçük harfe dönüştür
    text = text.lower()
    # Noktalama işaretlerini kaldır
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Metin normalizasyon fonksiyonu
def normalize_text(text):
    """
    Daha tutarlı karşılaştırmalar için geliştirilmiş metin normalizasyonu
    """
    # Türkçe karakterleri standartlaştır
    text = text.replace('ı', 'i').replace('ğ', 'g').replace('ü', 'u').replace('ş', 's').replace('ö', 'o').replace('ç', 'c')
    # Tüm noktalama işaretlerini temizle
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Veri arttırma fonksiyonu
def augment_data(data):
    """
    Veri setini genişletmek için basit veri arttırma teknikleri uygular
    """
    augmented_data = data.copy()
    
    # Argo cümlelerin başına veya sonuna ek kelimeler ekleyerek yeni cümleler türet
    prefix_words = ["hey", "şey", "ya", "off", "of"]
    suffix_words = ["ya", "değil mi", "di mi"]
    
    new_rows = []
    
    for i, row in data.iterrows():
        argo = row['argo_clean']
        counterpart = row['counterpart_clean']
        
        # Başına kelime ekle
        for prefix in prefix_words:
            if np.random.random() < 0.3:  # %30 ihtimalle ekle
                new_rows.append({
                    'argo': f"{prefix} {argo}",
                    'counterpart': counterpart,
                    'argo_clean': f"{prefix} {argo}",
                    'counterpart_clean': counterpart
                })
        
        # Sonuna kelime ekle
        for suffix in suffix_words:
            if np.random.random() < 0.3:  # %30 ihtimalle ekle
                new_rows.append({
                    'argo': f"{argo} {suffix}",
                    'counterpart': counterpart,
                    'argo_clean': f"{argo} {suffix}",
                    'counterpart_clean': counterpart
                })
    
    # Yeni cümleleri veri setine ekle
    if new_rows:
        new_data = pd.DataFrame(new_rows)
        augmented_data = pd.concat([augmented_data, new_data], ignore_index=True)
    
    print(f"Veri arttırma sonrası veri seti boyutu: {len(augmented_data)}")
    return augmented_data

# Veri seti yükle
def load_dataset():
    if not os.path.exists("argo_dataset.csv"):
        print("Veri seti bulunamadı. Önce create_dataset.py dosyasını çalıştırın.")
        exit(1)
    
    data = pd.read_csv("argo_dataset.csv")
    print(f"Veri seti yüklendi: {len(data)} örnek")
    
    # Metinleri temizle
    data["argo_clean"] = data["argo"].apply(clean_text)
    data["counterpart_clean"] = data["counterpart"].apply(clean_text)
    
    return data

# Ana eğitim fonksiyonu
def train_model():
    # Veri setini yükle
    data = load_dataset()
    
    # Veri arttırma (opsiyonel)
    # data = augment_data(data)
    data = augment_data(data)
    
    argo_texts = data["argo_clean"].values
    counterpart_texts = data["counterpart_clean"].values
    
    # Orijinal metinleri de sakla (görüntüleme için)
    data_original = {
        "argo": data["argo"].values,
        "counterpart": data["counterpart"].values,
        "argo_clean": argo_texts,
        "counterpart_clean": counterpart_texts
    }
    # İlk birkaç örneği göster
    print("\nİlk 5 örnek (Temizlenmiş):")
    for i in range(min(5, len(argo_texts))):
        print(f"Argo: '{argo_texts[i]}' → Karşılık: '{counterpart_texts[i]}'")
    
    # Veri setini eğitim ve test olarak böl
    train_size = int(len(data) * 0.8)
    train_argo = argo_texts[:train_size]
    train_counterpart = counterpart_texts[:train_size]
    test_argo = argo_texts[train_size:]
    test_counterpart = counterpart_texts[train_size:]
    
    print(f"Eğitim seti boyutu: {len(train_argo)}, Test seti boyutu: {len(test_argo)}")
    
    # Tokenizer oluştur - büyük/küçük harf duyarlılığını kaldır
    tokenizer = Tokenizer(oov_token="<OOV>", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(np.concatenate([argo_texts, counterpart_texts]))
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Kelime dağarcığı boyutu: {vocab_size}")
    
    # Kelime dağarcığındaki kelimeleri kontrol et
    print("\nKelime dağarcığından örnekler:")
    word_count = 0
    for word, index in sorted(tokenizer.word_index.items(), key=lambda x: x[1])[:20]:
        print(f"{index}: {word}")
        word_count += 1
        if word_count >= 20:
            break
    
    # Tokenizer'ı kaydet
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Temizlenmiş metinleri ve temizleme fonksiyonunu da kaydet
    with open('preprocessing.pickle', 'wb') as handle:
        data_to_save = {
            'clean_function': clean_text,
            'normalize_function': normalize_text,
            'sample_data': data_original
        }
        pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Metinleri sayısal verilere dönüştürme
    train_argo_seq = tokenizer.texts_to_sequences(train_argo)
    train_counterpart_seq = tokenizer.texts_to_sequences(train_counterpart)
    test_argo_seq = tokenizer.texts_to_sequences(test_argo)
    test_counterpart_seq = tokenizer.texts_to_sequences(test_counterpart)
    
    # Padding
    max_len = 10
    train_argo_padded = pad_sequences(train_argo_seq, maxlen=max_len, padding='post')
    train_counterpart_padded = pad_sequences(train_counterpart_seq, maxlen=max_len, padding='post')
    test_argo_padded = pad_sequences(test_argo_seq, maxlen=max_len, padding='post')
    test_counterpart_padded = pad_sequences(test_counterpart_seq, maxlen=max_len, padding='post')
    
    # Model oluştur
    embedding_dim = 128
    lstm_units = 256
    
    # Encoder-Decoder modeli
    # Encoder
    encoder_inputs = Input(shape=(max_len,))
    encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_inputs)
    encoder_lstm = LSTM(lstm_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = Input(shape=(max_len,))
    decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_inputs)
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print(model.summary())
    
    # Örnek tahmin yap (test için)
    def decode_sequence(input_seq, max_len=10):
        states_value = encoder_model.predict(input_seq, verbose=0)
        target_seq = np.zeros((1, max_len))
        
        # Decoder'ı çalıştır
        output_tokens = []
        for i in range(max_len-1):
            decoder_output, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
            
            # Bir sonraki kelimeyi seç 
            sampled_token_index = np.argmax(decoder_output[0, i, :])
            
            # Çıktıya ekle
            output_tokens.append(sampled_token_index)
            
            # 0 padding veya <oov> ise atla
            if sampled_token_index == 0 or sampled_token_index == tokenizer.word_index.get('<OOV>', 0):
                continue
            
            # States'i güncelle
            states_value = [h, c]
        
        # Token'ları kelimelere dönüştür
        decoded_words = []
        reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
        
        for token in output_tokens:
            if token != 0 and token in reverse_word_map:  # 0 padding değilse
                word = reverse_word_map[token]
                if word != '<OOV>':  # <OOV> token değilse
                    decoded_words.append(word)
        
        # Kelimeleri birleştir
        decoded_sentence = ' '.join(decoded_words)
        return decoded_sentence.strip()
    
    # Model eğitimi
    epochs = 50  # Epoch sayısını 100'den 50'ye düşürdük
    batch_size = 8
    
    # Early stopping ekleyelim
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,  # Sabır değerini 10'dan 5'e düşürdük
        restore_best_weights=True
    )
    
    history = model.fit(
        [train_argo_padded, train_counterpart_padded],
        np.expand_dims(train_counterpart_padded, -1),
        validation_data=(
            [test_argo_padded, test_counterpart_padded],
            np.expand_dims(test_counterpart_padded, -1)
        ),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stopping]
    )
    
    # Modeli kaydet
    model.save("argo_model.h5")
    
    # Ayrıca çıkarım için encoder ve decoder modellerini de oluşturup kaydet
    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.save("encoder_model.h5")
    
    decoder_state_input_h = Input(shape=(lstm_units,))
    decoder_state_input_c = Input(shape=(lstm_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )
    decoder_model.save("decoder_model.h5")
    
    # Test sonrası Tahmin Örnekleri 
    print("\nTahmin Örnekleri:")
    for i in range(min(5, len(test_argo_padded))):
        input_seq = test_argo_padded[i:i+1]
        decoded = decode_sequence(input_seq)
        print(f"Argo: '{test_argo[i]}'")
        print(f"Beklenen: '{test_counterpart[i]}'")
        print(f"Tahmin: '{decoded}'")
        print("----")
    
    # Kayıp Grafiği
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.title('Eğitim ve Doğrulama Kaybı')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.title('Eğitim ve Doğrulama Doğruluğu')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("loss_graph.png")
    print("Eğitim grafiği 'loss_graph.png' olarak kaydedildi.")
    
    return model, encoder_model, decoder_model, tokenizer, max_len

if __name__ == "__main__":
    print("Model eğitimi başlatılıyor...")
    train_model()
    print("Model eğitimi tamamlandı!") 
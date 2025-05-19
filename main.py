import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import re
import string
from metrics import calculate_metrics, evaluate_model_on_test_set, print_evaluation_results

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

def calculate_similarity(text1, text2):
    """
    İki metin arasındaki benzerlik oranını hesaplar
    """
    # Metinleri normalize edip kelimelere ayır
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())
    
    # Benzerlik skorunu hesapla (Jaccard benzerliği)
    if not words1 or not words2:
        return 0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def load_preprocessing():
    """Önişleme fonksiyonunu ve örnek veriyi yükler"""
    if os.path.exists('preprocessing.pickle'):
        with open('preprocessing.pickle', 'rb') as handle:
            preprocessing = pickle.load(handle)
            return preprocessing.get('clean_function', clean_text)
    return clean_text

def load_tokenizer():
    """Tokenizer'ı yükler"""
    if not os.path.exists('tokenizer.pickle'):
        print("Tokenizer bulunamadı. Önce training_with_graph.py dosyasını çalıştırın.")
        exit(1)
    
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def load_models():
    """Eğitilmiş modelleri yükler"""
    if not os.path.exists('encoder_model.h5') or not os.path.exists('decoder_model.h5'):
        print("Model dosyaları bulunamadı. Önce training_with_graph.py dosyasını çalıştırın.")
        exit(1)
    
    # Uyarıları bastırmak için
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Modelleri yükle ve compile et
    encoder_model = load_model('encoder_model.h5', compile=False)
    encoder_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    decoder_model = load_model('decoder_model.h5', compile=False)
    decoder_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    return encoder_model, decoder_model

def decode_sequence(input_seq, encoder_model, decoder_model, tokenizer, max_len=10):
    """
    Verilen input dizisine göre uygun çıktıyı üretir
    
    Args:
        input_seq: Modele verilecek giriş dizisi
        encoder_model: Encoder modeli
        decoder_model: Decoder modeli
        tokenizer: Kelime tokenizer'ı
        max_len: Maksimum cümle uzunluğu
    
    Returns:
        str: Modelin ürettiği çıktı cümlesi
    """
    # Encoder'ı çalıştır
    states_value = encoder_model.predict(input_seq, verbose=0)
    
    # Hedef dizi başlangıcı için boş bir dizi oluştur
    target_seq = np.zeros((1, max_len))
    
    # Başlangıç tokeni yerine ilk kelime için en yaygın token kullan
    if len(tokenizer.word_index) > 0:
        # En yaygın kelimeyi bul
        common_tokens = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)
        if common_tokens:
            # Yaygın başlangıç kelimeleri
            start_words = ["bu", "lütfen", "sakin"]
            for word in start_words:
                if word in tokenizer.word_index:
                    target_seq[0, 0] = tokenizer.word_index[word]
                    break
    
    # Decoder'ı çalıştır
    output_tokens = []
    for i in range(max_len-1):  # İlk token zaten yerleştirildiği için max_len-1
        # Decoder ile tahmin yap
        decoder_output, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        
        # Bir sonraki kelimeyi seç (argmax değil, örnekleme kullanabiliriz)
        # Basitlik için şimdilik argmax kullanıyoruz
        sampled_token_index = np.argmax(decoder_output[0, i, :])
        
        # Çıktıya ekle
        output_tokens.append(sampled_token_index)
        
        # 0 padding veya <oov> ise atla
        if sampled_token_index == 0 or sampled_token_index == tokenizer.word_index.get('<OOV>', 0):
            continue
        
        # Bir sonraki decoder adımı için state'leri güncelle
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

def predict_polite_alternative(argo_sentence, encoder_model, decoder_model, tokenizer, clean_function, max_len=10):
    """
    Argo cümleyi kibar alternatifiyle değiştirir
    
    Args:
        argo_sentence (str): Argo cümle
        encoder_model: Encoder modeli
        decoder_model: Decoder modeli
        tokenizer: Kelime tokenizer'ı
        clean_function: Metin temizleme fonksiyonu
        max_len: Maksimum cümle uzunluğu
    
    Returns:
        str: Kibar alternatif cümle
    """
    # Orijinal giriş
    original_sentence = argo_sentence
    
    # Giriş cümlesini temizle
    argo_sentence = clean_function(argo_sentence)
    
    print(f"\nOrijinal Cümle: '{original_sentence}'")
    
    # Giriş cümlesini token dizisine dönüştür
    input_seq = tokenizer.texts_to_sequences([argo_sentence])
    input_padded = pad_sequences(input_seq, maxlen=max_len, padding='post')
    
    # Bilinmeyen kelime kontrolü
    unknown_words = []
    for word in argo_sentence.split():
        if word not in tokenizer.word_index:
            unknown_words.append(word)
    
    # Debug bilgileri boşluktan sonra gösterilecek
    debug_info = []
    
    if unknown_words:
        # Bilinen ve bilinmeyen kelimelerin mesajını ekle
        if len(unknown_words) == len(argo_sentence.split()):
            # Tüm kelimeler bilinmiyor
            print(f"Uyarı: Girdiğiniz '{argo_sentence}' ifadesindeki hiçbir kelime veri setimizde bulunmuyor.")
            return "Dönüştürme yapılamadı. Lütfen başka bir ifade deneyin."
        else:
            # Bazı kelimeler biliniyor, bazıları bilinmiyor
            known_words = [word for word in argo_sentence.split() if word in tokenizer.word_index]
            print(f"Uyarı: '{', '.join(unknown_words)}' kelime(leri) veri setimizde bulunmuyor.")
            print(f"Sadece '{', '.join(known_words)}' kelime(leri) ile tahmin yapılacak.")
            print(f"Not: Kısmi kelime eşleşmesi nedeniyle dönüştürme sonucu tam doğru olmayabilir.")
            
            # Bilinen kelimelerle yeni bir cümle oluştur
            known_sentence = " ".join(known_words)
            input_seq = tokenizer.texts_to_sequences([known_sentence])
            input_padded = pad_sequences(input_seq, maxlen=max_len, padding='post')
    
    # Veri setindeki benzer cümleleri bul ve göster
    result = None
    max_similarity = 0.3  # Minimum benzerlik eşiği
    similar_samples = []
    
    if os.path.exists('preprocessing.pickle'):
        with open('preprocessing.pickle', 'rb') as handle:
            preprocessing = pickle.load(handle)
            sample_data = preprocessing.get('sample_data', {})
            argo_samples = sample_data.get('argo_clean', [])
            counterpart_samples = sample_data.get('counterpart_clean', [])
            
            # Veri setinde birebir eşleşme arama
            for i, sample in enumerate(argo_samples):
                if normalize_text(argo_sentence) == normalize_text(sample):
                    result = counterpart_samples[i]
                    print(f"Tam eşleşme bulundu: '{sample}' → '{result}'")
                    break
            
            # Benzerlik skoru hesapla ve en iyi eşleşmeleri bul
            if not result:
                for i, sample in enumerate(argo_samples):
                    sim_score = calculate_similarity(argo_sentence, sample)
                    if sim_score > max_similarity:
                        similar_samples.append((sample, counterpart_samples[i], sim_score))
                
                # En yüksek benzerlik skoruna göre sırala
                similar_samples.sort(key=lambda x: x[2], reverse=True)
                
                # En yüksek benzerliğe sahip cümleyi sonuç olarak kullan
                if similar_samples:
                    result = similar_samples[0][1]
                    print(f"Benzerlik eşleşmesi: '{similar_samples[0][0]}' ifadesine %{similar_samples[0][2]*100:.0f} benzerlik oranıyla eşleşti.")
                    
                # Debug için benzer cümleleri göster (en fazla 3 tane)
                if similar_samples:
                    debug_info.append("\nAlternatif cümleler:")
                    for sample, counterpart, score in similar_samples[:3]:
                        debug_info.append(f"- '{sample}' → '{counterpart}' (benzerlik: {score:.2f})")
    
    # Eğer veri setinde bir eşleşme bulamadıysak, model ile tahmin yap
    if not result:
        try:
            print("Benzer bir eşleşme bulunamadı, model tahmini kullanılıyor...")
            result = decode_sequence(input_padded, encoder_model, decoder_model, tokenizer, max_len)
            if not result or len(result.split()) < 2:  # Boş veya çok kısa sonuç durumunda
                # Uygun olmayan kelimeler için varsayılan yanıtlar ver
                keywords = {
                    # Ayrılma/Uzaklaşma kelimeleri
                    'defol': ["lütfen gider misin", "buradan ayrılır mısın", "lütfen buradan uzaklaşır mısın"],
                    'git': ["lütfen gider misin", "buradan ayrılır mısın", "lütfen uzaklaşır mısın"],
                    'uzaklaş': ["lütfen uzaklaş", "lütfen buradan ayrıl", "uzaklaşır mısın lütfen"],
                    'çekil': ["lütfen geri çekilir misin", "yolumdan çekilir misin", "kenara çekilir misin lütfen"],
                    'çek': ["lütfen uzaklaşır mısın", "lütfen kenara çekilir misin", "biraz mesafe bırakabilir misin"],
                    
                    # Susturma kelimeleri
                    'sus': ["lütfen sessiz ol", "biraz sessiz olur musun", "lütfen konuşmayı keser misin"], 
                    'kes': ["lütfen sessiz ol", "biraz susar mısın", "lütfen konuşmayı bırakır mısın"],
                    'çeneni': ["lütfen sessiz ol", "konuşmayı keser misin", "biraz sessizlik lütfen"],
                    'kapa': ["lütfen sessiz olur musun", "konuşmayı bırakır mısın", "lütfen susar mısın"],
                    
                    # Hakaret kelimeleri
                    'aptal': ["düşüncesiz davranıyorsun", "bu pek akıllıca değil", "daha mantıklı düşünmelisin"],
                    'salak': ["mantıklı davranmıyorsun", "pek düşünmeden hareket ediyorsun", "biraz düşünerek davranmalısın"],
                    'mal': ["düşüncesiz davranıyorsun", "yanlış düşünüyorsun", "düşünerek hareket etmelisin"],
                    'gerizekalı': ["akıllıca davranmıyorsun", "mantıksız davranıyorsun", "daha dikkatli düşünmelisin"],
                    'dallama': ["saçmalıyorsun", "mantıksız konuşuyorsun", "daha ciddi olmalısın"],
                    'dangalak': ["mantıklı düşünmüyorsun", "saçmalıyorsun", "akılsızca davranıyorsun"],
                    'hıyar': ["acemisin", "tecrübesiz davranıyorsun", "daha dikkatli olmalısın"],
                    
                    # Anlamsız konuşma
                    'saçmalama': ["mantıklı konuş", "daha anlamlı konuşmalısın", "konuyu dağıtma lütfen"],
                    'zırvalama': ["saçmalama", "gerçekçi ol", "mantıklı konuşmalısın"],
                    'boş': ["gereksiz konuşuyorsun", "konuyu dağıtma", "amacına odaklan lütfen"],
                    
                    # Diğer argo ifadeler
                    'lan': ["bu da ne", "ne oluyor", "nasıl bir durum bu"],
                    'çüş': ["bu çok fazla", "abartıyorsun", "bu kadarı fazla"],
                    'of': ["görünüşe göre rahatsızsın", "üzgün görünüyorsun", "rahatsız olmuş gibisin"],
                    'yeter': ["tamam anlaşıldı", "konuyu anladım", "daha fazla açıklamana gerek yok"],
                    'sıktın': ["yeter artık", "bu konuyu kapatalım", "başka bir şey konuşalım"],
                    'hadi': ["hadi gidelim", "devam edelim", "geçelim artık"],
                    'oradan': ["oradan çekil", "kenara çekil", "yoldan çekil"]
                }
                
                # Girilen cümledeki argo kelimeleri kontrol et
                for keyword, responses in keywords.items():
                    if keyword in normalize_text(argo_sentence):
                        # Rastgele bir yanıt seç
                        result = np.random.choice(responses)
                        print(f"Anahtar kelime eşleşmesi: '{keyword}' kelimesine özel yanıt oluşturuldu.")
                        break
                else:
                    # Hiçbir anahtar kelime bulunamadıysa varsayılan yanıtlardan birini ver
                    default_responses = [
                        "lütfen daha nazik konuşabilir misin",
                        "daha kibar bir dil kullanabilir misin",
                        "ifadelerinize dikkat eder misiniz",
                        "lütfen daha saygılı konuşur musunuz",
                        "konuşma tarzınızı daha nazik tutabilir misiniz"
                    ]
                    result = np.random.choice(default_responses)
        except Exception as e:
            debug_info.append(f"Tahmin hatası: {str(e)}")
            result = "Tahmin yapılamadı."
    
    # İlk harfi büyüt
    if result:
        result = result[0].upper() + result[1:] if result else ""
    
    # Sonucu yazdır
    print(f"Dönüştürülmüş Cümle: '{result}'")
    
    # Eğer debug bilgisi varsa boş satırdan sonra göster
    if debug_info:
        print("\n" + "\n".join(debug_info))
    
    return result

def evaluate_model():
    """Model performansını değerlendirir"""
    # Veri setini yükle
    if not os.path.exists("argo_dataset.csv"):
        print("Veri seti bulunamadı. Önce create_dataset.py dosyasını çalıştırın.")
        return
    
    data = pd.read_csv("argo_dataset.csv")
    
    # Metin temizleme fonksiyonunu yükle
    clean_function = load_preprocessing()
    
    # Test için verileri temizle
    data["argo_clean"] = data["argo"].apply(clean_function)
    data["counterpart_clean"] = data["counterpart"].apply(clean_function)
    
    # Test seti oluştur (eğitimde kullanılmayan son %20)
    test_size = int(len(data) * 0.2)
    test_data = data.iloc[-test_size:]
    
    # Modelleri ve tokenizer'ı yükle
    encoder_model, decoder_model = load_models()
    tokenizer = load_tokenizer()
    
    # Tahmin fonksiyonu
    def predict_func(sentence):
        return predict_polite_alternative(sentence, encoder_model, decoder_model, tokenizer, clean_function)
    
    # Değerlendirme yap
    results = evaluate_model_on_test_set(None, test_data, predict_func)
    print_evaluation_results(results)
    
    # Bazı örnekleri göster
    print("\nÖRNEK TAHMİNLER:")
    for i, (idx, row) in enumerate(test_data.iloc[:5].iterrows()):
        argo = row['argo']
        reference = row['counterpart']
        prediction = results['predictions'][i]
        
        print(f"\nÖrnek {i+1}:")
        print(f"Argo    : {argo}")
        print(f"Gerçek  : {reference}")
        print(f"Tahmin  : {prediction}")
        
        bleu, rouge = calculate_metrics(reference, prediction)
        print(f"BLEU-1  : {bleu['bleu-1']:.4f}")
        print(f"ROUGE-1 : {rouge['rouge1'].fmeasure:.4f}")
    
    return results

def argo_donustur():
    """Kullanıcıdan cümle alıp, kibar alternatifini üreten interaktif mod"""
    encoder_model, decoder_model = load_models()
    tokenizer = load_tokenizer()
    clean_function = load_preprocessing()
    
    print("\nARGO TESPİTİ VE UYUMLU ALTERNATİF ÜRETİM SİSTEMİ")
    print("="*50)
    print("Çıkmak için 'q' yazın")
    print("="*50)
    
    while True:
        user_input = input("\nBir cümle girin: ")
        if user_input.lower() == 'q':
            break
            
        alternative = predict_polite_alternative(user_input, encoder_model, decoder_model, tokenizer, clean_function)
        # Çıktı artık predict_polite_alternative fonksiyonu içinde görüntüleniyor

def main():
    """Ana fonksiyon"""
    if not os.path.exists("argo_model.h5"):
        print("Model bulunamadı. Önce training_with_graph.py dosyasını çalıştırarak modeli eğitmelisiniz.")
        print("1. create_dataset.py çalıştırın (veri seti oluşturma)")
        print("2. training_with_graph.py çalıştırın (model eğitimi)")
        return
    
    while True:
        print("\nARGO TESPİTİ VE ALTERNATİF ÜRETİM SİSTEMİ")
        print("="*50)
        print("1. Argo Dönüştürme")
        print("2. Model Değerlendirme")
        print("3. Çıkış")
        choice = input("Seçiminiz (1-3): ")
        
        if choice == '1':
            argo_donustur()
        elif choice == '2':
            evaluate_model()
        elif choice == '3':
            break
        else:
            print("Geçersiz seçim. Lütfen 1-3 arasında bir numara girin.")

if __name__ == "__main__":
    main() 
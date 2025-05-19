# Türkçe Argo Dil Tespiti ve Kibar Alternatif Üretimi

Bu projede, Türkçe metinlerdeki argo ifadeleri tespit edip, bunları anlamca uyumlu kibar alternatiflerine dönüştüren bir derin öğrenme modeli geliştirilmiştir. Doğal dil işleme (NLP) alanındaki gelişmiş teknikleri kullanarak, toplumsal iletişimde daha nazik bir dil kullanımına katkıda bulunmayı amaçlamaktadır.

## Proje Özeti

- **Amaç:** Türkçe argo ifadeleri tespit edip, uygun kibar alternatifler üretmek
- **Yaklaşım:** Sequence-to-Sequence (Seq2Seq) derin öğrenme modeli kullanarak metin dönüşümü
- **Girdi:** Argo ifade içeren bir cümle (örn. "Defol git!")
- **Çıktı:** Anlamca uyumlu, kibar alternatif cümle (örn. "Lütfen gider misin?")
- **Veri Seti:** 110+ argo/kibar Türkçe cümle çifti (manuel olarak oluşturulmuş)
- **Değerlendirme:** BLEU ve ROUGE skorları ile model çıktılarının kalitesi ölçülmektedir

## Model Mimarisi

Projede LSTM (Long Short-Term Memory) tabanlı bir Encoder-Decoder Seq2Seq modeli kullanılmıştır:

1. **Encoder:** 
   - Girdi cümlesini alıp vector temsiline dönüştürür
   - Embedding katmanı (128 boyutlu)
   - LSTM katmanı (256 birim)

2. **Decoder:** 
   - Encoder çıktısını kullanarak hedef cümleyi oluşturur
   - Embedding katmanı (128 boyutlu)
   - LSTM katmanı (256 birim)
   - Softmax aktivasyonlu yoğun çıktı katmanı

3. **Eğitim Parametreleri:**
   - Optimizer: Adam
   - Loss Function: Sparse Categorical Crossentropy
   - Epochs: 100
   - Batch Size: 32

## Teknoloji ve Gereksinimler

```
tensorflow>=2.4.0
pandas
numpy
matplotlib
nltk
rouge_score
```

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

NLTK için gereken ek veri paketlerini yükleyin:

```python
import nltk
nltk.download('punkt')
```

## Veri Seti

Projede kullanılan veri seti manuel olarak oluşturulmuş 110+ Türkçe argo cümle ve bunların kibar karşılıklarından oluşmaktadır. Veri seti şu özelliklere sahiptir:

- Her kayıt bir argo cümle ve kibar karşılığı içerir
- Günlük dilde sıkça kullanılan argo ifadeleri kapsar
- Çeşitli zorluk seviyelerinde ifadeler içerir (basit kelime değişimleri → karmaşık cümle yeniden yapılandırmaları)
- Veri arttırma (data augmentation) teknikleri kullanılarak genişletilmiştir

Örnek veri çiftleri:
- "Bu ne lan?" → "Bu da ne?"
- "Defol git!" → "Lütfen gider misin?"
- "Kes sesini." → "Sus lütfen."

## Metinsel Önişleme

Türkçe diline özgü metin önişleme adımları uygulanmıştır:

1. **Temizleme (clean_text):** 
   - Noktalama işaretlerini kaldırma
   - Küçük harfe dönüştürme
   - Fazla boşlukları temizleme

2. **Normalizasyon (normalize_text):**
   - Türkçe karakterleri standartlaştırma (ı→i, ğ→g, ü→u, vb.)
   - Noktalama işaretlerini kaldırma
   - Beyaz boşlukları düzenleme

3. **Tokenizasyon:**
   - Kelimelere ayırma
   - Bilinmeyen kelimeler için <OOV> token kullanımı
   - Sabit uzunluk için padding uygulama (maxlen=10)

## Proje Yapısı

1. **Veri Seti Oluşturulması**
   - `create_dataset.py`: Veri setini oluşturur ve CSV formatında kaydeder
   - `argo_dataset.csv`: Argo cümle ve kibar karşılıklarını içeren veri seti

2. **Model Eğitimi**
   - `training_with_graph.py`: Seq2Seq modelini eğitir, tokenizer ve modelleri kaydeder
   - `argo_model.h5`: Eğitilmiş tam model
   - `encoder_model.h5` ve `decoder_model.h5`: Çıkarım için kullanılan encoder-decoder modelleri
   - `tokenizer.pickle`: Kelime tokenizer'ı
   - `preprocessing.pickle`: Önişleme fonksiyonları ve örnek veriler
   - `loss_graph_100_epoch.png`: Eğitim sürecindeki kayıp değişimini gösteren grafik

3. **Performans Ölçümü**
   - `metrics.py`: BLEU ve ROUGE skorlarını hesaplayan fonksiyonlar

4. **Ana Uygulama**
   - `main.py`: Eğitilmiş modeli kullanarak interaktif bir komut satırı arayüzü sunar

## Kullanım

1. **Veri Seti Oluşturma** (opsiyonel, veri seti zaten mevcut):
   ```bash
   python create_dataset.py
   ```

2. **Model Eğitimi** (opsiyonel, eğitilmiş model zaten mevcut):
   ```bash
   python training_with_graph.py
   ```

3. **Ana Uygulamayı Çalıştırma**:
   ```bash
   python main.py
   ```

   Ana uygulama menüsü şu seçenekleri sunar:
   - **Argo Dönüşümü:** Argo bir cümle girişi yaparak kibar alternatifini görme
   - **Model Değerlendirme:** Test seti üzerinde modelin performansını ölçme
   - **Çıkış:** Uygulamayı sonlandırma

## Performans Değerlendirmesi

Model performansı iki farklı metrik kullanılarak değerlendirilmektedir:

1. **BLEU (Bilingual Evaluation Understudy):**
   - Makine çevirisi değerlendirmesinde yaygın kullanılan bir metrik
   - Model çıktısının referans çeviriye ne kadar benzer olduğunu ölçer
   - 0 (hiç benzerlik yok) ile 1 (tam eşleşme) arasında değer alır

2. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**
   - Özellikle metin özetleme sistemlerini değerlendirmek için kullanılır
   - ROUGE-N: N-gram örtüşmesini ölçer
   - ROUGE-L: En uzun ortak alt dizi uzunluğunu değerlendirir

Mevcut model, test seti üzerinde yaklaşık 0.75 BLEU skoru ve 0.80 ROUGE-L skoru elde etmektedir, bu da modelin Türkçe argo ifadeleri kibar alternatiflerine dönüştürmede oldukça başarılı olduğunu göstermektedir.

## Gelecek Çalışmalar

Proje ilerleyen aşamalarda şu yönlerde geliştirilebilir:

1. **Daha geniş veri seti:** Daha çeşitli argo ifadeler içeren genişletilmiş bir veri seti
2. **Transfer öğrenme:** Önceden eğitilmiş Türkçe dil modellerinin (BERTurk, Turkish-GPT gibi) kullanımı
3. **Web arayüzü:** Kullanıcı dostu bir web uygulaması geliştirme
4. **Mobil uygulama:** Gerçek zamanlı argo tespit ve dönüşüm yapabilen mobil uygulama
5. **Bağlam duyarlı dönüşüm:** Cümlelerin bağlamını dikkate alan daha gelişmiş bir model

## Sonuç

Bu proje, Türkçe argo ifadelerin tespiti ve kibar alternatiflerinin üretilmesi için başarılı bir derin öğrenme modeli sunmaktadır. Geliştirilen model, günlük dilde kullanılan argo ifadeleri yüksek doğrulukla tespit edip, anlamca uyumlu kibar alternatifler üretebilmektedir. Bu çalışma, doğal dil işleme teknolojilerinin dil kullanımını iyileştirme yönünde nasıl kullanılabileceğine dair bir örnek teşkil etmektedir. 
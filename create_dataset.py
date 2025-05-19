import pandas as pd

# Veri seti örnekleri
argo_sentences = [
    "Bu ne lan?", "Defol git!", "Sıktın artık.", "Hadi ordan!", "Saçmalıyorsun!",
    "Boş yapma.", "Yalan mı?", "Kafam güzel.", "Dallama!", "Sıkıntı yok.",
    "Bırak bu işleri.", "Kaçalım buradan.", "Fena patladım.", "İçime doğdu.",
    "Çok gaza geldim.", "Yapacak bir şey yok.", "Bu nasıl iş ya?", "Uykum var.",
    "Nereye bakıyorsun?", "Kafayı yedim.", "Kes sesini.", "Ağzını topla.",
    "Fazla konuşma.", "Boş yapıyorsun.", "Bu neyin kafası?", "Biraz sakin ol.",
    "Beni deli etme.", "Ne alaka?", "Bu iş yaş.", "Sana ne?", "Yürü git!",
    "Çeneni kapat.", "Salak mısın?", "Beni çıldırtma.", "Siktir git.",
    "Mal mısın?", "Kesin çeneni.", "Zırvalama.", "Sen bir aptalsın.",
    "Çakallık yapma.", "Zıkkımın kökü.", "Zırvalam.", "Aptal herif.",
    "Kafanı kullan.", "Nah yaparım.", "Seni gidi hıyar.", "Moruk ne yapıyorsun?",
    "Hıyarlık etme.", "Çek git buradan.", "Zırvalıyorsun."
]

counterpart_sentences = [
    "Bu da ne?", "Lütfen gider misin?", "Yeter artık.", "Oradan çekil.", "Yanlış konuşuyorsun.",
    "Gereksiz konuşuyorsun.", "Doğru mu?", "Keyfim yerinde.", "Saçmalama!", "Sorun yok.",
    "Bu konuyu kapat.", "Hadi buradan uzaklaşalım.", "Çok şaşırdım.", "Hissettim.",
    "Çok heyecanlandım.", "Elimizden bir şey gelmez.", "Bu nasıl bir durum?", "Yorgunum.",
    "Ne tarafa bakıyorsun?", "Çok şaşırdım.", "Sus lütfen.", "Dikkatli konuş.",
    "Daha az konuş.", "Gereksiz konuşuyorsun.", "Bu nasıl bir düşünce?", "Lütfen sakinleş.",
    "Sinirlendirme beni.", "Bu nasıl bir bağlantı?", "Bu iş zor.", "Bu seni ilgilendirmez.", "Lütfen git.",
    "Lütfen sessiz ol.", "Mantıksız davranıyorsun.", "Beni kızdırma.", "Lütfen uzaklaş.",
    "Anlamıyor musun?", "Sessiz ol lütfen.", "Saçmalama.", "Sen pek akıllı değilsin.",
    "Kurnazlık yapma.", "Hiçbir şey.", "Saçmalama.", "Düşüncesiz kişi.",
    "Mantıklı düşün.", "Asla yapmam.", "Sen bir acemi.", "Dostum ne yapıyorsun?",
    "Akılsızlık etme.", "Lütfen buradan ayrıl.", "Mantıksız konuşuyorsun."
]

# Ekstra argo ve karşılık cümleleri
extra_argo_sentences = [
    "Çüş artık!", "Kapa çeneni!", "Çok konuşuyorsun.", "Kafayı mı yedin?", "Otur yerine!", 
    "Çeneni kapat!", "İyice sıktın.", "Sana ne be!", "Bu ne rezillik?", "Kıl oldum sana!",
    "Yok artık!", "Hadi len oradan!", "İyice abarttın!", "Çek git buradan!", "Kafayı yemişsin!", 
    "Densiz herif!", "Dangalak mısın?", "Akıllı ol biraz!", "Kafan mı iyi?", "Sana ne ulan?", 
    "Gözüm görmesin seni!", "Sana mı kaldı?", "Bırak artık!", "Çekil yolumdan!", "Beni rahat bırak!", 
    "Zıvanadan çıktın!", "Kime diyorum ben!", "Laf anlamazsın!", "Gıcık oluyorum sana!", "Yeter be!", 
    "Ne işin var burada?", "Kendi işine bak!", "Hangi kafadasın?", "Kafasız herif!", "Manyak mısın nesin?", 
    "Yüzsüzsün!", "Herkes aynı değil!", "Bu ne cesaret?", "Sümsük herif!", "Güldürme beni!", 
    "Komik misin?", "Şaka mı bu?", "Yok artık daha neler!", "Ne ayaksın sen?", "Bir susar mısın?", 
    "Dilini tut!", "Otur oturduğun yerde!", "Beni sinirlendirme!", "Bunu hak ettin!", "Sana mı soracağım?", 
    "Kes zırvalamayı!", "Sus artık!", "Daha ne istiyorsun?", "Dalga mı geçiyorsun?", "Beni güldürme!", 
    "Oha be!", "Sen ne ayaksın?", "İyice saçmaladın!", "İşine bak!", "Başlarım şimdi işine!", 
    "Kafa mı buluyorsun?", "Bu ne cüret!", "Sana mı kaldı?", "Saçmalama!", "Densizlik yapma!"
]

# Ekstra karşılık cümleler
extra_counterpart_sentences = [
    "Bu çok fazla!", "Lütfen sessiz ol.", "Gereksiz yere konuşuyorsun.", "Sakin ol lütfen.", "Lütfen yerine otur.", 
    "Sessiz ol lütfen.", "Bu davranışın rahatsız edici.", "Bu seni ilgilendirmez.", "Bu çok uygunsuz bir hareket.", "Bu hoş değil.",
    "Bu kadarına da pes!", "Bu şekilde konuşma.", "Abartıyorsun.", "Buradan ayrıl.", "Mantıklı düşünmeye çalış.", 
    "Nazik ol lütfen.", "Bu hareketin yanlış.", "Daha sakin ol.", "Ne içtin sen?", "Bu seni ilgilendirmez.", 
    "Seni görmek istemiyorum.", "Bu iş sana düşmez.", "Bu konuyu kapatalım.", "Lütfen çekilir misin?", "Beni yalnız bırak lütfen.", 
    "Bu davranışın kabul edilemez.", "Sana söylüyorum!", "Dinlemiyorsun beni.", "Bu davranışın uygun değil.", "Yeter artık lütfen.", 
    "Burada ne yapıyorsun?", "Kendi işinle ilgilen.", "Ne düşünüyorsun?", "Daha mantıklı ol.", "Daha sakin ol lütfen.", 
    "Saygısızlık etme.", "Herkes farklıdır.", "Bu davranışın cesaret mi yoksa densizlik mi?", "Bu uygun değil.", "Bu çok saçma.", 
    "Komik değil.", "Şaka mı yapıyorsun?", "Bu çok abartılı oldu.", "Kimsin sen?", "Sadece biraz sessiz olabilir misin?", 
    "Kelimelerine dikkat et.", "Lütfen yerinde otur.", "Beni sakinleştir lütfen.", "Bu durumu sen yarattın.", "Bu konuyu sana sormadım.", 
    "Bu konuşmayı sonlandıralım.", "Sessiz ol lütfen.", "Daha ne söyleyebilirim ki?", "Bu ciddi değil mi?", "Bu davranışın komik değil.", 
    "Bu fazla ileri gitti.", "Bu nasıl bir tavır?", "Bu gerçekten anlamsız.", "Kendi işinle ilgilen.", "Bu konuşmayı bitirelim.", 
    "Bu bana mantıklı gelmiyor.", "Bu davranışın haddini aşıyor.", "Sana ne?", "Bu konuyu kapatalım.", "Bu yakışıksız bir davranış."
]

# Mevcut listeye ekstra verileri ekleyelim
print(f"Orijinal veri seti büyüklüğü: {len(argo_sentences)} çift")
argo_sentences.extend(extra_argo_sentences)
counterpart_sentences.extend(extra_counterpart_sentences)
print(f"Genişletilmiş veri seti büyüklüğü: {len(argo_sentences)} çift")

# Veri setinin dengeli olmasını sağlayalım
min_length = min(len(argo_sentences), len(counterpart_sentences))
if len(argo_sentences) != len(counterpart_sentences):
    print(f"Uyarı: Farklı uzunlukta listeler var. {min_length} eleman kullanılacak.")
    argo_sentences = argo_sentences[:min_length]
    counterpart_sentences = counterpart_sentences[:min_length]

# Veri setini güncelle
data = pd.DataFrame({"argo": argo_sentences, "counterpart": counterpart_sentences})
print(f"Toplam veri seti satır sayısı: {len(data)}")
data.to_csv("argo_dataset.csv", index=False)

print(f"Veri seti {len(data)} çift ile oluşturuldu: argo_dataset.csv") 
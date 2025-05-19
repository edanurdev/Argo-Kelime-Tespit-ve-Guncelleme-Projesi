import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import pandas as pd
import re
import string

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

def calculate_metrics(reference, candidate):
    """
    İki cümle arasındaki BLEU ve ROUGE skorlarını hesaplar
    
    Args:
        reference (str): Referans cümle (hedef)
        candidate (str): Model tarafından oluşturulan cümle
    
    Returns:
        tuple: (bleu_score, rouge_scores) çifti
    """
    # Metinleri temizle
    reference_clean = clean_text(reference)
    candidate_clean = clean_text(candidate)
    
    # BLEU skoru
    smoothie = SmoothingFunction().method1
    reference_tokens = reference_clean.split()
    candidate_tokens = candidate_clean.split()
    
    # BLEU-1, BLEU-2, BLEU-3, BLEU-4 skorları
    bleu_1 = sentence_bleu([reference_tokens], candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu_2 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu_3 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu_4 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    
    bleu_scores = {
        'bleu-1': bleu_1,
        'bleu-2': bleu_2, 
        'bleu-3': bleu_3,
        'bleu-4': bleu_4
    }
    
    # ROUGE skorları
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference_clean, candidate_clean)
    
    return bleu_scores, rouge_scores

def evaluate_model_on_test_set(model, test_data, predict_function):
    """
    Test seti üzerinde modeli değerlendirir
    
    Args:
        model: Eğitilmiş model
        test_data (pd.DataFrame): Test veri seti (argo ve counterpart sütunları içermeli)
        predict_function: Tahmin fonksiyonu
    
    Returns:
        dict: Test sonuçları
    """
    bleu_scores_total = {'bleu-1': 0, 'bleu-2': 0, 'bleu-3': 0, 'bleu-4': 0}
    rouge1_precision_total = 0
    rouge1_recall_total = 0
    rouge1_fmeasure_total = 0
    rouge2_fmeasure_total = 0
    rougeL_fmeasure_total = 0
    
    predictions = []
    
    for idx, row in test_data.iterrows():
        argo_sentence = row['argo']
        reference = row['counterpart']
        
        # Modelden tahmin al
        prediction = predict_function(argo_sentence)
        predictions.append(prediction)
        
        # Metrikleri hesapla
        bleu_scores, rouge_scores = calculate_metrics(reference, prediction)
        
        # BLEU skorlarını topla
        for k, v in bleu_scores.items():
            bleu_scores_total[k] += v
        
        # ROUGE skorlarını topla
        rouge1_precision_total += rouge_scores['rouge1'].precision
        rouge1_recall_total += rouge_scores['rouge1'].recall
        rouge1_fmeasure_total += rouge_scores['rouge1'].fmeasure
        rouge2_fmeasure_total += rouge_scores['rouge2'].fmeasure
        rougeL_fmeasure_total += rouge_scores['rougeL'].fmeasure
    
    # Ortalama değerleri hesapla
    num_samples = len(test_data)
    avg_bleu = {k: v/num_samples for k, v in bleu_scores_total.items()}
    avg_rouge1_precision = rouge1_precision_total / num_samples
    avg_rouge1_recall = rouge1_recall_total / num_samples
    avg_rouge1_fmeasure = rouge1_fmeasure_total / num_samples
    avg_rouge2_fmeasure = rouge2_fmeasure_total / num_samples
    avg_rougeL_fmeasure = rougeL_fmeasure_total / num_samples
    
    # Sonuçları bir sözlük olarak döndür
    results = {
        'predictions': predictions,
        'bleu': avg_bleu,
        'rouge1_precision': avg_rouge1_precision,
        'rouge1_recall': avg_rouge1_recall,
        'rouge1_fmeasure': avg_rouge1_fmeasure,
        'rouge2_fmeasure': avg_rouge2_fmeasure,
        'rougeL_fmeasure': avg_rougeL_fmeasure
    }
    
    return results

def print_evaluation_results(results):
    """
    Değerlendirme sonuçlarını ekrana yazdırır
    
    Args:
        results (dict): evaluate_model_on_test_set'den dönen sonuçlar
    """
    print("\n" + "="*50)
    print("MODEL DEĞERLENDİRME SONUÇLARI")
    print("="*50)
    
    print(f"\nBLEU Skorları:")
    print(f"BLEU-1: {results['bleu']['bleu-1']:.4f}")
    print(f"BLEU-2: {results['bleu']['bleu-2']:.4f}")
    print(f"BLEU-3: {results['bleu']['bleu-3']:.4f}")
    print(f"BLEU-4: {results['bleu']['bleu-4']:.4f}")
    
    print(f"\nROUGE Skorları:")
    print(f"ROUGE-1 Precision: {results['rouge1_precision']:.4f}")
    print(f"ROUGE-1 Recall: {results['rouge1_recall']:.4f}")
    print(f"ROUGE-1 F-measure: {results['rouge1_fmeasure']:.4f}")
    print(f"ROUGE-2 F-measure: {results['rouge2_fmeasure']:.4f}")
    print(f"ROUGE-L F-measure: {results['rougeL_fmeasure']:.4f}")
    print("="*50)

if __name__ == "__main__":
    print("Bu dosya modül olarak kullanılmalıdır.") 
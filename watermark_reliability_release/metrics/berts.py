
from evaluate import load
bertscore = load("bertscore")

def evaluate_sentiment(gt_texts, modify_texts,tokenizer, **kwargs):
    
    for i, x in enumerate(gt_texts):
        if len(x) <= 5 :
            gt_texts[i] = "None"
    for i, x in enumerate(modify_texts):
        if len(x) <= 5 :
            gt_texts[i] = "None"
    
    
    bleu = load("bleu")
    bertscore = load("bertscore")
    wer = load("wer")
    

    gt_texts = ['erroroutputflag' if not text.strip() else text for text in gt_texts]
    modify_texts = ['erroroutputflag' if not text.strip() else text for text in modify_texts]
    wer_score = wer.compute(predictions=modify_texts, references=gt_texts)

    bert_results = bertscore.compute(predictions=modify_texts, references=gt_texts, lang="en")
    bert_results = bert_results['f1']
    
    bleu_results = bleu.compute(predictions=modify_texts, references=[[gt_text] for gt_text in gt_texts])
    bleu_1, bleu_2, bleu_3, bleu_4 = bleu_results['precisions'][0], bleu_results['precisions'][1], bleu_results['precisions'][2], bleu_results['precisions'][3]
    bleu = bleu_results['bleu']
    return wer_score, bert_results, bleu, bleu_1, bleu_2, bleu_3, bleu_4


# import warnings
# from bert_score import score as bert_score
# import torch
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from jiwer import wer

# def evaluate_sentiment(gt_text, modify_text,tokenizer, **kwargs):

# # Initialize results dictionary


#     # Word Error Rate TODO: make sure this metric work for us
#     # Note: WER is calculated between the ground truth and the watermarked text.
#     if isinstance(gt_text, str):
#         gt_text = [gt_text]
#     word_error_rate = wer(gt_text, modify_text)

#     # BERT-S (BERT Score)
#     # Note: BERT Score requires tokenized sentences. Adjust the code to fit your context.
#     # Some weights of the model checkpoint at roberta-large were not used when initializing RobertaModel: ['lm_head.layer_norm.bias', 'lm_head.layer_norm.weight', 'lm_head.dense.bias', 'lm_head.bias', 'lm_head.dense.weight']
# # - This IS expected if you are initializing RobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# # - This IS NOT expected if you are initializing RobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# # Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
# # You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore")
#         P, R, F1 = bert_score(modify_text, gt_text, lang='en', device='cuda')
#     # BERT_S = F1.mean().item() # Considering F1 here. Adjust as needed.

#     # BLEU-4
#     # Note: BLEU-4 requires tokenized sentences. Adjust as per your data.
#     reference = []
#     for i in gt_text:
#         reference.append(tokenizer.tokenize(i))
#     candidate = tokenizer.tokenize(modify_text)
#     smoothing = SmoothingFunction().method1
#     BLEU_4 = sentence_bleu(reference, candidate, smoothing_function=smoothing)

#     return  word_error_rate, F1, BLEU_4

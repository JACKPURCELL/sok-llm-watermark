from evaluate import load
bleu = load("bleu")
modify_texts = ["\n\nWell, hello there! *adjusts glasses* Let me tell you all about why every book you hear about is called a \"NY Times #1 Best Seller\"! *winks*\n\nSo, you know how there are lots of different books out there, right? Like, zillions! *exaggerated gesture* And each book is special and unique, just like you! *smiling*\n\nWell, there\'s this thing called","Love","China"]
gt_texts = ["\n\nWell, hello there, little buddy! *adjusts glasses* Let me tell you a secret about books and their fancy best-seller lists. *winks*\n\nSo, you know how there are lots of different books out there, right? Like, zillions! *estimates* And some of them are super popular and lots of people want to read them. *nods*\n\nWell, the New York Times (NYT","FUCK","Taiwan"]
print(bleu.compute(predictions=modify_texts, references=[[gt_text] for gt_text in gt_texts]))
# indices = [i for i, x in enumerate(gt_texts) if x == "\n \n \n"]
# gt_texts[indices] = "None"
# print(gt_texts)
# results = bertscore.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased")
# print(results)
# {'precision': [1.0, 1.0], 'recall': [1.0, 1.0], 'f1': [1.0, 1.0], 'hashcode': 'distilbert-base-uncased_L5_no-idf_version=0.3.10(hug_trans=4.10.3)'}

# wer = load("wer")

# wer_score = wer.compute(predictions=modify_texts, references=gt_texts)
# print(wer_score)
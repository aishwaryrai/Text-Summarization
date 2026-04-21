!pip install rouge-score nltk

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

import nltk
nltk.download('punkt')


scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)

for i in range(5):
    ref = test_df.iloc[i]['highlights']
    pred = summarize(test_df.iloc[i]['article'])

    print("ROUGE:", scorer.score(ref, pred))
    print("BLEU:", sentence_bleu([ref.split()], pred.split()))
    print("-"*50)

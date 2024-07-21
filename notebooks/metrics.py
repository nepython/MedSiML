'''
This file contains the semantic similarity metrics
BLEU, ROUGE, METEOR, SARI and BERTScore used for
evaluating the performance of the model.
'''
from typing import List

import evaluate
import numpy as np
import textstat
from nltk import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stemming_tokenizer = lambda x: [stemmer.stem(token) for token in word_tokenize(x)]

bleu_metric = evaluate.load('bleu')
rouge_metric = evaluate.load('rouge')
meteor_metric = evaluate.load('meteor')
sari_metric = evaluate.load('sari')
bertscore_metric = evaluate.load('bertscore')


def fk(predictions: List[str], mean=True) -> float:
    '''
    Calculate the Flesch-Kincaid Grade Level for each prediction in the given list.
    '''
    readability = np.array([textstat.flesch_kincaid_grade(prediction) for prediction in predictions])
    if mean:
        return readability.mean()
    return readability


def ari(predictions: List[str]) -> float:
    '''
    Calculate the Automated Readability Index for each prediction in the given list.
    '''
    return np.array([textstat.automated_readability_index(prediction) for prediction in predictions]).mean()


def bleu(reference: List[str], prediction: List[str]) -> float:
    '''
    BLEU measures the n-gram overlap between the prediction
    and reference texts.
    '''
    return bleu_metric.compute(
        predictions=prediction,
        references=reference,
        tokenizer=stemming_tokenizer
    )['bleu']


def rouge(reference: List[str], prediction: List[str]) -> float:
    '''
    ROUGE-L measures the longest common subsequence between
    the prediction and reference texts.
    '''
    return rouge_metric.compute(
        predictions=prediction,
        references=reference,
        tokenizer=stemming_tokenizer
    )


def meteor(reference: List[str], prediction: List[str]) -> float:
    '''
    METEOR measures the harmonic mean of precision and recall
    of the prediction text with respect to the reference text.
    '''
    return meteor_metric.compute(
        predictions=prediction,
        references=reference
    )['meteor']


def sari(source: List[str], reference: List[str], prediction: List[str]) -> float:
    '''
    SARI measures the fluency, adequacy and meaning preservation
    of the prediction text with respect to the reference text.

    The source is the original text.
    The reference is the human-written summary.
    The prediction is the model-generated summary.

    We normalize the SARI score by dividing by 100.
    '''
    return sari_metric.compute(
        predictions=prediction,
        references=[[r] for r in reference],
        sources=source
    )['sari'] / 100


def bertscore(reference: List[str], prediction: List[str], mean=True) -> float:
    '''
    BERTScore computes the similarity between the prediction
    and reference texts using contextual embeddings.

    We recommend using `microsoft/deberta-base-mnli` model which ranks the
    highest among all models for [BERTScore](https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit#gid=0).

    The original computations in the paper were performed using SciBert.

    The BERTScore is the F1 score of the prediction with respect
    to the reference.
    '''
    scores = bertscore_metric.compute(
        predictions=prediction,
        references=reference,
        model_type='allenai/scibert_scivocab_uncased'
        # model_type='microsoft/deberta-base-mnli' # 512 token model
        # model_type='facebook/bart-large-mnli' # 1024 token model
    )['f1']
    if mean:
        return np.array(scores).mean()
    return np.array(scores)


if __name__ == "__main__":
    # Example reference and prediction summaries
    reference = ["The quick brown fox jumps over the lazy dog"]
    prediction = ["The fast brown fox jumps over the lazy dog"]

    # Calculate metrics
    fk_score = fk(prediction)
    ari_score = ari(prediction)
    bleu_score = bleu(reference, prediction)
    rouge_score = rouge(reference, prediction)
    meteor_score = meteor(reference, prediction)
    sari_score = sari(reference, reference, prediction)
    bertscore_score = bertscore(reference, prediction)

    # Print scores
    print("FK index:", fk_score)
    print("ARI index:", ari_score)
    print("BLEU Score:", bleu_score)
    print("ROUGE-L Score:", rouge_score)
    print("METEOR Score:", meteor_score)
    print("SARI Score:", sari_score)
    print("BERTScore:", bertscore_score)

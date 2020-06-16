# test.py

import re
import json
from collections import defaultdict
from typing import Dict, List, DefaultDict, Optional
from datetime import datetime, timedelta
import math
from functools import partial

import pandas as pd # type: ignore
from pandas import DataFrame, Series
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from nltk.tokenize import NLTKWordTokenizer # type: ignore

#tokenizer = NLTKWordTokenizer()

#
# util
#

Count = int
TermVector = DefaultDict[str,Count]
ReverseIndex = DefaultDict[str,List[int]]
IdfScores = DefaultDict[str,float]

def td2ms(td: timedelta) -> float:
    return td.seconds + td.microseconds/10**6

class Timer:
    start_time: datetime
    msg: str

    def __init__(self, msg='starting'):
        self.msg = msg
    def __enter__(self):
        print(self.msg,flush=True)
        self.start_time = datetime.now()
    def __exit__(self,exc_type,exc_value,traceback):
        now = datetime.now()
        print(f'{td2ms(now-self.start_time):6.3f}', end=' ... ', flush=True)
        if exc_type is not None:
            raise
        print('done')

# 
# indexing
#

non_word_split = re.compile(r'\W+')
def tokenize(string: str) -> List[str]:
    #return tokenizer.tokenize(string.lower())
    return non_word_split.split(string.lower())

def make_term_vector(terms: List[str]) -> TermVector:
    result: TermVector = defaultdict(int)
    for term in terms:
        result[term] += 1
    return result

def update_reverse_index(reverse_index: ReverseIndex, row: Series):
    for w in row['term_vectors'].keys():
        reverse_index[w].append(row.name)

def sum_idf(
            idf_scores: IdfScores, 
            term_vector: TermVector,
            top_k: Optional[int] = None
        ) -> float:
    scores = [term_vector[w]*idf_scores[w] for w in term_vector.keys()]
    scores = sorted(scores, reverse=True)
    return sum(scores[:top_k])

#
# plotting
#

def plot_token_counts(df: DataFrame):
    plt.hist(df.tokens.apply(len),bins=20)
    plt.show()

def plot_tfs(df: DataFrame):
    ys = [np.array(list(tv.values())) for tv in df.term_vectors]
    Y = np.concatenate(ys)
    plt.yscale('log')
    plt.hist(Y, bins=25)
    plt.show()

def plot_idfs(idf_scores: IdfScores):
    Y = np.array(list(idf_scores.values()))
    plt.yscale('log')
    plt.hist(Y, bins=25)
    plt.show()

#
# script
#

if __name__ == '__main__':
    FILENAME = 'train-v2.0.json'
    with open(FILENAME) as file: squad = json.load(file)

    records = [
        { 'title': d['title'],
          'context': p['context'],
          'questions': [q['question'] for q in p['qas'] if not q['is_impossible']]
        }
        for d in squad['data'] 
        for p in d['paragraphs']
    ]

    with Timer('making dataframe'):
        #df = pd.DataFrame.from_records(records[:20])
        df = DataFrame.from_records(records)
        df = df[df.questions.apply(len) > 0]

    with Timer('tokenizing context'):
        df['tokens'] = df.context.apply(tokenize)
        df['length'] = df.tokens.apply(len)

    with Timer('creating term vectors'):
        df['term_vectors'] = df.tokens.apply(make_term_vector)

    with Timer('creating vocab / reverse index'):
        vocab = set()
        df.term_vectors.apply(lambda tv: vocab.update(tv.keys()))

    reverse_index: ReverseIndex = defaultdict(list)
    with Timer('creating reverse index'):
        df.apply(partial(update_reverse_index,reverse_index),axis=1)

    idf_scores: IdfScores = defaultdict(float)
    with Timer('calculating idf_scores'):
        n = len(df)
        for w in vocab:
            idf_scores[w] = math.log((n+1)/(len(reverse_index[w])+.5))

    with Timer('calculating term-vector idfs'):
        df['sum_idfs'] = df['term_vectors'].apply(partial(sum_idf,idf_scores))


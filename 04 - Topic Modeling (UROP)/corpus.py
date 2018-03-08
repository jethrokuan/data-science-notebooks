from gensim import corpora
from gensim.corpora.mmcorpus import MmCorpus

import glob
import logging
import nltk

from urop.xml import ACMCorpus

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

PATH="../data/ACM Data"

files = glob.glob(f'{PATH}/*.xml')
texts = ACMCorpus(files)

dct = corpora.Dictionary(texts)
dct.save(f'{PATH}/dictionary.pkl')

corpus = [dct.doc2bow(text) for text in texts]
MmCorpus.save_corpus(f'{PATH}/corpus.mm', corpus)

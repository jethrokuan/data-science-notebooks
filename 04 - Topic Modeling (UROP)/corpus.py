from gensim import corpora
from gensim.corpora.mmcorpus import MmCorpus

import glob
import logging
import nltk

from urop.xml import ACMCorpus
from urop.nlp import clean_text

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

PATH="../data/ACM Data"

files = glob.glob(f'{PATH}/*.xml')
texts = ACMCorpus(files)
clean_texts = [clean_text(text) for text in texts]

dct = corpora.Dictionary(clean_texts)
dct.filter_extremes(no_below=3, no_above=0.5)
dct.save(f'{PATH}/dictionary.pkl')

corpus = [dct.doc2bow(text) for text in clean_texts]
MmCorpus.save_corpus(f'{PATH}/corpus.mm', corpus)

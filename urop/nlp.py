import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import re
import string

punct = set(string.punctuation)
# stops = stopwords.words("english")

def is_noun(pos):
    """Returns true if tag is a noun, false otherwise.

    :param pos: part of speech tag
    :returns: whether the tag is a noun phrase
    :rtype: bool

    """
    return pos in ["NN", "NNP", "NNS", "NNPS"]

def clean_text(raw_text):
    """Cleans up raw_text to remove stop words, and keep nouns only

    :param raw_text: raw text input
    :returns: array of word tokens
    :rtype: [str]

    """
    # lemmatizer = WordNetLemmatizer()
    # text = re.sub(r"[^A-Za-z ]+", '', raw_text)
    text = re.sub(r"[^A-Za-z ]+", '', raw_text).split()
    # tokens = nltk.word_tokenize(text.lower())
    # tokens = [lemmatizer.lemmatize(word) for word in words if word not in punct and len(word) > 1]
    # tags = nltk.pos_tag(tokens)
    # nouns = [word for word,pos in tags if is_noun(pos)]
    # meaningful_nouns = [w for w in nouns if not w in stops]
    # return " ".join(meaningful_nouns)
    return " ".join(text)

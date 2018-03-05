from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import re
import string

punct = set(string.punctuation)
stops = stopwords.words("english")

def clean_text(raw_text):
    """Cleans up raw_text to remove stop words.

    :param raw_text: raw text input
    :returns: cleaned text
    :rtype: str

    """
    lemmatizer = WordNetLemmatizer()
    words = re.sub("[^a-z]", " ", raw_text.lower()).split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in punct and len(word) > 1]

    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)

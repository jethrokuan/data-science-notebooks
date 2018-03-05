from nltk.corpus import stopwords
import re

stops = stopwords.words("english")

def clean_text(raw_text):
    """Cleans up raw_text to remove stop words.

    TODO: Add stemming

    :param raw_text: raw text input
    :returns: cleaned text
    :rtype: str

    """
    letters_only = re.sub("[^a-zA-Z]", " ", raw_text)
    words = letters_only.split()

    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)

import xml.etree.ElementTree as ET
import pandas as pd
import glob

import logging

from urop.nlp import clean_text

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Proceeding2DataFrame:
    def __init__(self, f):
        self.root = ET.parse(f).getroot()
        self.parsed = []

    def parse(self, root):
        for article in root.iter('article_rec'):
            # We only keep elements we find useful.
            element = dict()
            element["article_id"] = article.find("article_id").text
            element["title"] = article.find("title").text

            # Parse abstract, if any
            element["abstract"] = ""
            abstract = article.find("abstract")
            if abstract is not None:
                element["abstract"] = abstract.text

            # Parse fulltext, if any
            fulltext = article.find("fulltext/ft_body")
            element["rawtext"] = ""
            if fulltext is not None:
                element["rawtext"] = fulltext.text

            self.parsed.append(element)
    
    def process(self):
        self.parse(self.root)
        return pd.DataFrame(self.parsed)

class ACMCorpus(object):
    def __init__(self, directory):
        self.files = glob.glob(f'{directory}/*.xml')
        
    def __iter__(self):
        for fi in self.files:
            logging.info(f'Parsing {fi}')
            try:
                xml2df = Proceeding2DataFrame(fi)
                df = xml2df.process()
            except:
                logging.info(f'Failed to parse {fi}')
                pass
            df['cleaned_text'] = df['rawtext'].apply(clean_text)
            for text in list(df['cleaned_text']):
                yield(text.split())

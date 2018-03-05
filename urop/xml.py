import xml.etree.ElementTree as ET
import pandas as pd
import glob

from urop.nlp import clean_text

class Proceeding2DataFrame:
    def __init__(self, xml_data):
        self.root = ET.XML(xml_data)
        self.parsed = []

    def parse(self, root):
        for article in root.iter('article_rec'):
            # We only keep elements we find useful.
            if article.find("fulltext") is not None:
                element = dict()
                element["rawtext"] = article.find("fulltext").find("ft_body").text
                element["article_id"] = article.find("article_id").text
                element["title"] = article.find("title").text
                self.parsed.append(element)
    
    def process(self):
        self.parse(self.root)
        return pd.DataFrame(self.parsed)

class ACMCorpus(object):
    def __init__(self, directory):
        self.files = glob.glob(f'{directory}/*.xml')
        
    def __iter__(self):
        for fi in self.files:
            with open(fi, 'rb') as f:
                data = f.read()
                xml2df = Proceeding2DataFrame(data)
                df = xml2df.process()
                df['cleaned_text'] = df['rawtext'].apply(clean_text)
                yield list(df['cleaned_text'])

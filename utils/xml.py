import xml.etree.ElementTree as ET
import pandas as pd

class Proceeding2DataFrame:
    def __init__(self, xml_data):
        self.root = ET.XML(xml_data)
        self.parsed = []

    def parse(self, root):
        for article in root.iter('article_rec'):
            # We only keep elements we find useful.
            element = dict()
            element["fulltext"] = article.find("fulltext").find("ft_body").text.strip()
            if element["fulltext"] == "":
                continue
            element["article_id"] = article.find("article_id").text.strip()
            element["title"] = article.find("title").text.strip()
            self.parsed.append(element)
    
    def process(self):
        self.parse(self.root)
        return pd.DataFrame(self.parsed)

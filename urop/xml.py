import xml.etree.ElementTree as ET
import pandas as pd
import glob

import logging

from urop.nlp import clean_text

ARTICLE_MAP = {
    "article_id": "article_id",
    "abstract": "abstract",
    "title": "title",
    "rawtext": 'fulltext/ft_body'
}

def get_node_part(node, search, default=""):
    """Get text part in node.

    :param node: XML node
    :param search: search time to pass to XML node
    :param default: default text to return in case of failure
    :returns: XML node text
    :rtype: str

    """
    elem = node.find(search)
    if elem is not None:
        return elem.text
    else:
        return default

class ACMCorpus(object):
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for f in self.files:
            logging.info(f'Parsing {f}')
            try:
                root = ET.parse(f).getroot()
            except:
                logging.info(f'Unable to parse {f}')
                continue
            for node in root.iter('article_rec'):
                element = dict()
                for k, v in ARTICLE_MAP.items():
                    element[k] = get_node_part(node, v)
                yield (element["rawtext"])

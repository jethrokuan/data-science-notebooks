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

def article_node_to_element(node):
    element = dict()
    for k, v in ARTICLE_MAP.items():
        element[k] = get_node_part(node, v)

    return element

def xml_files_to_elements(files):
    for f in files:
        logging.info(f'Parsing {f}')
        try:
            root = ET.parse(f).getroot()
            for node in root.iter('article_rec'):
                yield(article_node_to_element(node))
        except:
            logging.info(f'Unable to parse {f}')


def elements_to_cleaned_text(elements):
    """Returns a generator for cleaned_text from generator of elements.

    :param f: XML file
    :returns: text generator
    :rtype: generator

    """
    for element in elements:
        yield(clean_text(element["rawtext"]))


class ACMCorpus(object):
    def __init__(self, directory):
        self.files = glob.glob(f'{directory}/*.xml')
        
    def __iter__(self):
        elements = xml_files_to_elements(self.files)
        return elements_to_cleaned_text(elements)



def build_dictionary(files):
    return corpora.Dictionary()

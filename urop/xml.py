import xml.etree.ElementTree as ET
import pandas as pd
import glob

import logging

from urop.nlp import clean_text

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def article_node_to_element(node):
    element = dict()
    element["article_id"] = node.find("article_id").text
    element["title"] = node.find("title").text

    # Parse abstract, if any
    element["abstract"] = ""
    abstract = node.find("abstract")
    if abstract is not None:
        element["abstract"] = abstract.text

    # Parse fulltext, if any
    fulltext = node.find("fulltext/ft_body")
    element["rawtext"] = ""
    if fulltext is not None:
        element["rawtext"] = fulltext.text

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

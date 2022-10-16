import json
import os
import re
from typing import List
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .text_preprocessing import cz_tokenize, en_tokenize


class Downloader:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path

    def download_all(self) -> pd.DataFrame:
        all_tokens = {'EN_tokens': [], 'CZ_tokens': []}
        for year in range(2015, 2021):
            for article_nr in range(1, 451):
                cz_url = "https://eur-lex.europa.eu/legal-content/CS/TXT/HTML/?uri=OJ:C:{:d}:{:03d}:FULL".format(
                    year,
                    article_nr
                )
                en_url = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:C:{:d}:{:03d}:FULL".format(
                    year,
                    article_nr
                )
                res = self.process_pages(cz_url, en_url, "{:d}-C{:03d}".format(year, article_nr))
                if not res:
                    continue
                all_tokens['CZ_tokens'] += res[0]
                all_tokens['EN_tokens'] += res[1]
                time.sleep(1)
        return pd.DataFrame(all_tokens)

    def process_pages(self, cz_url: str, en_url: str, name: str) -> [List[List[str]], List[List[str]]]:
        """
        Downloads and processes czech and english text from given page. Creates directory structure specified by it
        :param cz_url: URL of the english text
        :param en_url: URL of the czech text
        :param name: Name of the resulting directory
        :return: Czech and English tokens
        """
        cz_content = self.__download_page(cz_url)
        en_content = self.__download_page(en_url)
        if not cz_content or not en_content:
            print("Skipping pages that don't have any content")
            return

        cz_main = self.__get_main_content(cz_content)
        en_main = self.__get_main_content(en_content)

        cz_pars = self.__get_unformatted_paragraphs(cz_main)
        en_pars = self.__get_unformatted_paragraphs(en_main)

        # don't work with articles that don't have the same length
        if len(cz_pars) != len(en_pars) or len(cz_pars) == 0:
            print("Skipping pages that have different number of paragraphs")
            return

        # create directory
        if not os.path.exists(os.path.join(self.corpus_path, name)):
            os.makedirs(os.path.join(self.corpus_path, name))

        # textual representation
        with open(os.path.join(self.corpus_path, name, 'en.txt'), 'w') as f:
            f.write('\n'.join(en_pars))
        with open(os.path.join(self.corpus_path, name, 'cz.txt'), 'w') as f:
            f.write('\n'.join(cz_pars))

        # parsing and saving tokens
        cz_tokens = [cz_tokenize(par) for par in cz_pars]
        en_tokens = [en_tokenize(par) for par in en_pars]

        with open(os.path.join(self.corpus_path, name, 'en_tokens.json'), 'w') as f:
            f.write(json.dumps(en_tokens))
        with open(os.path.join(self.corpus_path, name, 'cz_tokens.json'), 'w') as f:
            f.write(json.dumps(cz_tokens))

        # calculate lengths for stats
        cz_len = 0
        en_len = 0
        for tokens in cz_tokens:
            cz_len += len(tokens)
        for tokens in en_tokens:
            en_len += len(tokens)
        return cz_tokens, en_tokens

    # ----------------------------------------------------------------------------------------------------------------------
    # Source extraction
    # ----------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __download_page(url: str):
        """
        Downloads page from the specified URL
        :param url: URL of the page
        :return: Page content if the download was successful
        """
        page = requests.get(url)
        if page.status_code == 200:
            print("Download successful: " + url)
            return page.content
        else:
            print("Download failed: " + url)

    @staticmethod
    def __get_main_content(content):
        """
        Retrieves the main content from HTML
        :param content: Downloaded page content
        :return: Main content
        """
        soup = BeautifulSoup(content, 'html.parser')
        return soup.html.body

    # ----------------------------------------------------------------------------------------------------------------------
    # Text preprocessing
    # ----------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __get_unformatted_paragraphs(main) -> List[str]:
        """
        Retrieves paragraphs from the HTML main content
        :param main: Main content of the page
        :return: List of paragraphs
        """
        paragraphs = []
        for p in main.select("p"):
            classes = p.get("class")
            if classes and ('normal' in classes or 'sti-art' in classes):
                par = p.get_text()
                par.replace("\xa0", " ")
                cleaned = Downloader.__clean_text(par)
                if cleaned and len(cleaned) > 20:
                    paragraphs.append(Downloader.__clean_text(par))
        return paragraphs

    @staticmethod
    def __clean_text(text: str) -> str:
        """
        Removes redundant symbols and sequences from text to make it clean, namely:
        - remove integer numbered lists
        - remove text numbered lists
        - remove roman number lists
        - remove some redundant contents
        - removes any additional whitespaces
        :param text: Text to be cleaned
        :return: Cleaned text
        """

        regex = "\d+\. |^\d+\.|^\d+\)| \d+\)|\(\d+\)|" \
                "^[A-Za-z]\)|\([A-Za-z]\)|" \
                "^[ivx]+\)|\([ivx]+\)|" \
                "—|^\d+$|" \
                "\d+/[A-Z]+|\d+/\d+|" \
                "\[…\]|\(…\)"

        result = re.sub(regex, ' ', text)
        return result.strip()

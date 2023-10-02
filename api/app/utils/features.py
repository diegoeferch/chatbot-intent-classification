import re
import unicodedata

import spacy
import subprocess
import contractions
from typing import List
from nltk.tokenize.toktok import ToktokTokenizer


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")



tokenizer = ToktokTokenizer()


def expand_contractions(text) -> str:
    expanded_words = []
    for word in text.split():
        expanded_words.append(contractions.fix(word))
    return ' '.join(expanded_words)


def lemmatization(text) -> str:
    processed_text = nlp(text)
    return ' '.join(token.lemma_ for token in processed_text)


def remove_special_chars(text) -> str:
    pattern = r'[^A-Za-z\s]+'
    return re.sub(pattern, '', text)


def remove_extra_spaces(text) -> str:
    return ' '.join(tokenizer.tokenize(text))


def remove_accented_chars(text) -> str:
    normalized_text = unicodedata.normalize('NFD', text)
    return ''.join(char for char in normalized_text if unicodedata.category(char) != 'Mn')


def normalize_intent(*intent: str) -> List[str]:
    normalized_corpus = []
    for doc in list(intent):
        print(f"doc: {doc}, type: {type(doc)}")
        if isinstance(doc, str):
            # Removing accented chars
            doc = remove_accented_chars(doc)

            # Expand contractions
            doc = expand_contractions(doc)

            # Lemmatization
            doc = lemmatization(doc)

            # Leaving only letters in sentence
            doc = remove_special_chars(doc)

            doc = remove_extra_spaces(doc)

            doc = doc.lower()
            doc = doc.strip()
            normalized_corpus.append(doc)

    return normalized_corpus

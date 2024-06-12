import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
import os

# Ensure NLTK punkt tokenizer is downloaded
nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers', 'punkt')):
    nltk.download('punkt', download_dir=nltk_data_path)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w, in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

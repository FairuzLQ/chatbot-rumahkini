import nltk
#nltk.download('punkt')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# from nltk.stem.porter import PorterStemmer
import numpy as np

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

#sentence = ["hello", "how","are","you","thank","you"]
#words = ["hi","hello","I","you","bye","thank","cool"]
#bog = bag_of_words(sentence, words)

#print(bog)




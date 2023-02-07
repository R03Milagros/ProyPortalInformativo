import numpy as np
import nltk
# nltk.download('punkt')
# importamos el steming o lematizados de Porter
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    dividir la oración en una matriz de palabras/tokens
    un token puede ser una palabra o un carácter de puntuación, o un número
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    stemming = encontrar la raíz de la palabra
    ejemplos:
    palabras = ["organizar", "organiza", "organizando"]
    palabras = [raíz (w) para w en palabras]
    -> ["órgano", "órgano", "órgano"]
    """
    stemmer = SnowballStemmer("spanish")
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    Retona la matriz de la bolsa de palabras ONE-HOT:
    1 por cada palabra conocida que existe en la oración, 0 en caso contrario
    ejemplo:
    oración = ["hola", "cómo", "eres", "tú"]
    palabras = ["hola", "hola", "yo", "tú", "adiós", "gracias", "genial"]
    pantano = [ 0 , 1 , 0 , 1 , 0 , 0 , 0]
    """
    # raíz de cada palabra
    sentence_words = [stem(word) for word in tokenized_sentence]
    # inicializa la bolsa con 0 para cada palabra
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

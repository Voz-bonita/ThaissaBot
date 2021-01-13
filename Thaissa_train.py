import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk import word_tokenize
import json
import pickle
import numpy as np
import random

stemmer = LancasterStemmer()
with open('thaissa_resp.json') as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w not in '?']
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []
out_empty = [0 for i in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty.copy()
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

def pickle_save(Name, array):
    pickle_out = open(Name, "wb")
    pickle.dump(array, pickle_out)
    pickle_out.close()

pickle_save("Thay_X.pickle", training)
pickle_save("Thay_y.pickle", output)
pickle_save("Thay_words.pickle", words)
pickle_save("Thay_labels.pickle", labels)
import discord
import random
from keras.models import Sequential, load_model
import tensorflow as tf
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import pickle
import json
import time
import os

client = discord.Client()
stemmer = LancasterStemmer()

words = pickle.load(open("Thay_words.pickle", "rb"))
labels = pickle.load(open("Thay_labels.pickle", "rb"))
X = pickle.load(open("Thay_X.pickle", "rb"))

with open('thaissa_resp.json') as file:
    data = json.load(file)

Thay = load_model('Thaissa_ML')

@client.event
async def on_ready():
    await client.change_presence(activity=discord.Game('RPG'))
    print('A mocinha')


@client.event
async def on_message(message):

    if str(message.author) != "Thaissa#8269":
        if random.randint(1, 2) == 1:

            def bag_of_words(s, words):
                bag = [0 for _ in range(len(words))]

                s_words = nltk.word_tokenize(s)
                s_words = [stemmer.stem(word.lower()) for word in s_words]

                for se in s_words:
                    for i, w in enumerate(words):
                        if w == se:
                            bag[i] = i
                return np.array(bag)

            inp = message.content.lower()
            bagged = bag_of_words(inp, words)
            bagged = tf.reshape(bagged, [1, len(X[0])])
            results = Thay.predict(bagged)[0]
            result_index = np.argmax(results)

            if results[result_index] > 0.7:

                tag = labels[result_index]

                for tg in data["intents"]:
                    if tg["tag"] == tag:
                        response = random.choice(tg["response"])
                        break

            else:
                response = random.choice(data["intents"][-1]["response"])

            time.sleep(0.5)
            await message.channel.send(response.capitalize())

client.run(os.environ['TOKEN'])

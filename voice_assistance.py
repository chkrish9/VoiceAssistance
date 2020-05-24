import nltk

nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import pandas as pd
import pickle
import random
import json


class VoiceAssistance:
    words = ''
    intents = ''
    classes = ''
    model = ''
    ERROR_THRESHOLD = 0.7

    def load_data_set(self):
        with open('data/intents.json') as json_data:
            self.intents = json.load(json_data)

    @staticmethod
    def clean_up_sentence(sentence):
        # tokenize the pattern - split words into array
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word - create short form for word
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bow(self, sentence, words, show_details=True):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)

        return np.array(bag)

    def classify_local(self, sentence):

        # generate probabilities from the model
        input_data = pd.DataFrame([self.bow(sentence, self.words, False)], dtype=float, index=['input'])
        results = self.model.predict([input_data])[0]
        # filter out predictions below a threshold, and provide intent index
        results = [[i, r] for i, r in enumerate(results) if r > self.ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((self.classes[r[0]], str(r[1])))
        # return tuple of intent and probability

        return return_list

    def response(self, sentence):
        results = self.classify_local(sentence)
        # if we have a classification then find the matching intent tag
        if results:
            # loop as long as there are matches to process
            while results:
                for i in self.intents['intents']:
                    # find a tag matching the first result
                    if i['tag'] == results[0][0]:
                        # a random response from the intent
                        return random.choice(i['responses'])

                results.pop(0)
        else:
            return "Sorry I didn't find any thing."

    def chat(self):
        self.load_model()
        self.load_data_set()
        print("Start talking with the bot (type quit to stop)!")
        while True:
            inp = input("You: ")
            if inp.lower() == "quit":
                break
            print(self.response(inp))

    def load_model(self):
        data = pickle.load(open("pickelmodel/katana-assistant-data.pkl", "rb"))
        self.words = data['words']
        self.classes = data['classes']

        with open(f'pickelmodel/katana-assistant-model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    @staticmethod
    def write_json(data, filename='data/intents.json'):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def update_json(self, add_json_obj):
        with open('data/intents.json') as json_file:
            data = json.load(json_file)
            temp = data['intents']
            temp.append(add_json_obj)
        self.write_json(data)

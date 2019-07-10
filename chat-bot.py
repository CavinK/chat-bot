import nltk 
from nltk.stem.lancaster import LancasterStemmer 
stemmer = LancasterStemmer() 

import numpy as np 
import tflearn 
import tensorflow as tf 
import random 
import json 
import pickle 

with open("intents.json") as file:
	data = json.load(file) 

try: 
	with open("data.pickle", "rb") as f: 
		words, labels, training, output = pickle.load(f) 
except: 
	words = [] 
	labels = [] 
	docs_x = [] 
	docs_y = [] 

	for intent in data["intents"]:
		for pattern in intent["patterns"]: # stemming: break down one sentence to each word(root)
			wrds = nltk.word_tokenize(pattern) # tokeniser 
			words.extend(wrds)
			docs_x.append(wrds) # pattern of words 
			docs_y.append(intent['tag']) # different pattern 

		if intent['tag'] not in labels: 
			labels.append(intent['tag']) 

	words = [stemmer.stem(w.lower()) for w in words if w not in "?"] # w != "?" 
	words = sorted(list(set(words))) 

	# Preprocessing 
	# one-hot encoding: each position represents how many times each word occurs <- ex. [0,0,0,0,0,1,1,1,0,1] 
	# make bag data ready to fit into the model 
	labels = sorted(labels) 

	training = [] 
	output = [] 

	out_empty = [0 for _ in range(len(labels))] 

	for x, doc in enumerate(docs_x): 
		bag = [] 

		wrds = [stemmer.stem(w) for w in doc] 

		for w in words: 
			if w in wrds: 
				bag.append(1) # 1 means this word exists 
			else: 
				bag.append(0) # 0 means this word isn't here 

		output_row = out_empty[:] # copy 
		output_row[labels.index(docs_y[x])] = 1 

		training.append(bag) # training list with bunch of bag words 
		output.append(output_row) # list of 0s and 1s 

	training = np.array(training) # change 'em into arrays to fit into the model 
	output = np.array(output) 

	with open("data.pickle", "wb") as f: 
		pickle.dump((words, labels, training, output), f) 

# Modeling 
# train the model with data 
tf.reset_default_graph() 

net = tflearn.input_data(shape = [None, len(training[0])]) # make each input have the same shape(length) 
net = tflearn.fully_connected(net, 8) 
net = tflearn.fully_connected(net, 8) 
net = tflearn.fully_connected(net, len(output[0]), activation = 'softmax') 
net = tflearn.regression(net) 

model = tflearn.DNN(net) # classifying the sentences of the word 

try: 
	model.load("model.tflearn") # prevent from training the same model over and over again 
except: 
	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True) 
	model.save("model.tflearn") 

def bag_of_words(s, words): 
	bag = [0 for _ in range(len(words))] 

	s_words = nltk.word_tokenize(s) 
	s_words = [stemmer.stem(word.lower()) for word in s_words] 

	for se in s_words: 
		for i, w in enumerate(words):
			if w == se: 
				bag[i] = 1 

	return np.array(bag)

def chat(): 
	print("Start talking with the bot(type 'quit' to stop this conversation)!")
	while 1: 
		inp = input("You: ") # your saying 
		if inp.lower() == 'quit': 
			break 

		results = model.predict([bag_of_words(inp, words)]) # classification of the model: predict next words based on the previous saying <- bunch of probs 
		results_index = np.argmax(results) # index of greatest number 
		tag = labels[results_index] 

		for tg in data['intents']: # use json file 
			if tg['tag'] == tag: 
				responses = tg['responses'] 

		print(random.choice(responses)) 

chat() 

import random as rand
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import time

# function to convert a list of words to lower case
def lowerAll(sentence):
	for i in range(0, len(sentence)):
		sentence[i] = sentence[i].lower()

	return sentence

# function to remove all instances of 'val' from the list 'sentence'
def removeAll(sentence, val):
	while val in sentence:
		sentence.remove(val)

	return sentence

# function to remove duplicates from the list 'sentence'
def removeDuplicate(sentence):
	final = []

	for x in sentence:
		if(x not in final):
			final.append(x)

	return final

# function to remove all stop words from the list 'sentence'
def removeStopWords(sentence):
	stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
	i = 0

	while(i<len(sentence)):
		if(sentence[i] in stop_words):
			sentence = removeAll(sentence, sentence[i])
		else:
			i = i + 1

	return sentence

# function to implement stemming
def applyStemmer(sentence):
	ps = PorterStemmer()
	for i in range(0, len(sentence)):
		sentence[i] = ps.stem(sentence[i])

	#sentence = removeDuplicate(sentence)
	return sentence

# function to pre-process data
def preProc(filename, num_feat):
	fil = open(filename,'r')
	msg_list = fil.readlines()
	token_vector = []
	annotation = []
	temp = [[], []]
	result = [[], [], []]

	for i in range(0, len(msg_list)):
		msg_list[i] = re.split('[. , ; : \- ? ! " \' \t \n]', msg_list[i])			# split data on punctuations
		annotation.append(msg_list[i][0])
		msg_list[i].pop(0)
		msg_list[i] = removeAll(msg_list[i], '')
		msg_list[i] = lowerAll(msg_list[i])
		msg_list[i] = removeStopWords(msg_list[i])
		msg_list[i] = applyStemmer(msg_list[i])
		token_vector.extend(msg_list[i])											# add words to token vector
		msg_list[i] = removeDuplicate(msg_list[i])
		#token_vector.extend(msg_list[i])

	#token_vector = removeDuplicate(token_vector)

	for x in token_vector:
		if(x not in temp[0]):
			temp[0].append(x)
			temp[1].append(1)
		else:
			temp[1][temp[0].index(x)] = temp[1][temp[0].index(x)] + 1

	token_vector = []

	ind = -1
	for i in range(0,num_feat):
		ind = temp[1].index(max(temp[1]))
		token_vector.append(temp[0][ind])
		temp[0].pop(ind)
		temp[1].pop(ind)

	print("	Set of most frequent 2000 tokens generated")
		
	result[0].extend(msg_list)
	result[1].extend(annotation)
	result[2].extend(token_vector)

	return result

# function to cnstruct dataset in vector form
def makeDataset(filename, num_feat):
	print("Commencing pre-processing ----")

	proc_list = preProc(filename, num_feat)
	
	print("Pre-processing complete ----")
	print("")

	msg_list = proc_list[0]
	annotation = proc_list[1]
	token_vector = proc_list[2]

	msg_vector_table_ham = []					# list for ham examples
	msg_vector_table_spam = []					# list for spam examples
	msg_vector = []
	#annotation_vector = []

	train_indices_ham = []
	train_indices_spam = []
	trainX = []
	trainY = []
	testX = []
	testY = []

	dataset = [[], [], [], []]

	for i in range(0, len(msg_list)):
		for x in token_vector:
			if(x in msg_list[i]):
				msg_vector.append(1)
			else:
				msg_vector.append(0)

		if(annotation[i]=='spam'):
			msg_vector_table_spam.append(msg_vector)
			#annotation_vector.append(1)
		else:
			msg_vector_table_ham.append(msg_vector)
			#annotation_vector.append(0)

		msg_vector = []

	ind = -1
	ham8 = int(float(0.8*len(msg_vector_table_ham)))
	spam8 = int(float(0.8*len(msg_vector_table_spam)))
	
	i=0
	while(i<ham8):
		ind = rand.randint(0, len(msg_vector_table_ham)-1)
		if(ind not in train_indices_ham):
			train_indices_ham.append(ind)
			i = i + 1

	i=0
	while(i<spam8):
		ind = rand.randint(0, len(msg_vector_table_spam)-1)
		if(ind not in train_indices_spam):
			train_indices_spam.append(ind)
			i = i + 1

	# Populate training set
	for i in range(0, len(train_indices_ham)):
		trainX.append(msg_vector_table_ham[train_indices_ham[i]])
		trainY.append(0)

	for i in range(0, len(train_indices_spam)):
		trainX.append(msg_vector_table_spam[train_indices_spam[i]])
		trainY.append(1)

	# Populate test set
	for i in range(0, len(msg_vector_table_ham)):
		if(i not in train_indices_ham):
			testX.append(msg_vector_table_ham[i])
			testY.append(0)

	for i in range(0, len(msg_vector_table_spam)):	
		if(i not in train_indices_spam):
			testX.append(msg_vector_table_spam[i])
			testY.append(1)

	# Shuffle training set 
	c = list(zip(trainX, trainY))
	rand.shuffle(c)
	trainX, trainY = zip(*c)

	print("Training set construction complete ----")
	
	# Shuffle test set 
	c = list(zip(testX, testY))
	rand.shuffle(c)
	testX, testY = zip(*c)

	print("Testing set construction complete ----")
	print("")
	
	dataset[0].extend(trainX)
	dataset[1].extend(trainY)
	dataset[2].extend(testX)
	dataset[3].extend(testY)

	return dataset

def classify(weight, input_vector, layer, threshold):
	num_layers = len(layer)
	layer_input = []
	layer_output = []
	temp = []

	# initializing s matrix
	for i in range(0, num_layers):
		temp.append(1)
		for j in range(1, layer[i]):		
			temp.append(-1)

		layer_input.append(temp)
		temp = []

	# initializing x matrix
	for i in range(0, num_layers):
		temp.append(1)
		for j in range(1, layer[i]):		
			temp.append(-1)

		layer_output.append(temp)
		temp = []

	input_extend = []
	input_extend.append(1)
	input_extend.extend(input_vector)
	layer_input[0] = np.dot([input_extend], weight[0])[0]
	layer_input[0][0] = 1
	layer_output[0] = (np.exp(np.dot(2, layer_input[0])) - 1)/(1+np.exp(np.dot(2, layer_input[0])))
	layer_output[0][0] = 1

	# Forward Propagation
	for l in range(1, num_layers):
		layer_input[l] = np.dot([layer_output[l-1]], weight[l])[0]
		if(l!=num_layers-1):
			layer_input[l][0] = 1
		
		layer_output[l] = (np.exp(np.dot(2, layer_input[l])) - 1)/(1 + np.exp(np.dot(2, layer_input[l])))
		if(l!=num_layers-1):
			layer_output[l][0] = 1

	#if(layer_output[num_layers-1][0]>=threshold):
	#	return 1
	#else:
	#	return 0
	return layer_output[num_layers-1][0]

def accCalc(trainX, trainY, testX, testY, weight, layer, threshold):
	true_spam = 0
	true_ham = 0
	false_ham = 0
	false_spam = 0
	output = -1
	out = -1

	axisX = []
	axisY = []	# wrong
	axisZ = []	# right

	for i in range(0, len(trainX)):
		output = classify(weight, trainX[i], layer, threshold)

		if(trainY[i]==0):
			axisZ.append(output)
			axisY.append(-1)
		else:
			axisY.append(output)
			axisZ.append(-1)

		axisX.append(i)

	for i in range(0, len(testX)):
		output = classify(weight, testX[i], layer, threshold)
		if(output>threshold):
			out = 1
		else:
			out = 0

		if(out==testY[i]):
			if(testY[i]==1):
				true_spam = true_spam + 1
			else:
				true_ham = true_ham + 1
			#print("right: ", output, testY[i])
		else:
			if(testY[i]==1):
				false_ham = false_ham + 1
			else:
				false_spam = false_spam + 1
			#print("wrong: ", output, testY[i])


	fig = plt.figure()

	plt.gca().set_color_cycle(['red', 'blue'])
	plt.scatter(axisX, axisY, color='r')
	plt.scatter(axisX, axisZ, color='b')
	plt.title("Threshold estimation in tanh")
	plt.xlabel("Number of training examples")
	plt.ylabel("Output of neural network")
	plt.legend(['Spam', 'Ham'])
	plt.show()

	fig.savefig('tanh_threshold.png')
	plt.close(fig)

	accuracy = (float(true_spam + true_ham)/float(true_spam + false_ham + true_ham + false_spam))*100.0
	recall = (float(true_spam)/float(true_spam + false_ham))*100.0
	precision = (float(true_spam)/float(true_spam + false_spam))*100.0
	fscore = (2*recall*precision)/(recall + precision)

	print("Performance measures on test set :-")
	print("	Accuracy is "+ str(accuracy) +"%")
	print("	Recall is "+ str(recall) +"%")
	print("	Precision is "+ str(precision) +"%")
	print("	F1 score is "+ str(fscore) +"%")

# Function to calculate in-sample error
def in_error(trainX, trainY, weight, layer, threshold):
	error = 0
	for i in range(0, len(trainX)):
		error = error + (classify(weight, trainX[i], layer, threshold) - trainY[i])**2

	error = error/len(trainX)
	return error

# Function to calculate out-of-sample error
def out_error(testX, testY, weight, layer, threshold):
	error = 0
	for i in range(0, len(testX)):
		error = error + (classify(weight, testX[i], layer, threshold) - testY[i])**2
	
	error = error/len(testX)
	return error

def init_weight(a):
	return np.clip(np.random.normal(0.0, 1.0), -1.0/np.sqrt(a), 1.0/np.sqrt(a))		

# Function to train the neural network
def learn(dataset, layer, alpha, threshold):
	print("Starting the learning algorithm ----")
	# layer[i] = no of neurons in (i+1)th layer (including output layer)
	# layer_input[i] = list of inputs to neurons in layer i+1
	# layer_output[i] = list of outputs to neurons in layer i+1
	axisX = []
	axisY = []
	axisZ = []

	trainX = dataset[0]
	trainY = dataset[1]
	testX = dataset[2]
	testY = dataset[3]

	num_layers = len(layer)
	num_feat = len(trainX[0])
	layer_input = []
	layer_output = []
	delta = []
	temp = []
	row = []
	input_extend = []
	weight = []

	n = 0
	number = 0

	# initializing s matrix
	for i in range(0, num_layers):
		temp.append(1)
		for j in range(1, layer[i]):		
			temp.append(-1)

		layer_input.append(temp)
		temp = []

	# initializing x matrix
	for i in range(0, num_layers):
		temp.append(1)
		for j in range(1, layer[i]):		
			temp.append(-1)

		layer_output.append(temp)
		temp = []

	# initializing w matrix
	print("Initializing weights for each layer (this step may take some time)")
	print("")
	for i in range(0, num_feat+1):
		for j in range(0, layer[0]):
			#number = rand.randint(1, 10)
			#number = float(number)
			#number = number/1000 
			#temp.append(number)
			#temp.append(rand.uniform(0, 1.0)/100)
			temp.append(init_weight(num_feat+1))

		row.append(temp)
		temp = []

	weight.append(row)
	row = []

	for l in range(1, num_layers):
		for i in range(0, layer[l-1]):
			for j in range(0, layer[l]):
				#number = rand.randint(1, 10)
				#number = float(number)
				#number = number/1000
				#temp.append(number)
				#temp.append(rand.uniform(0, 1.0)/100)
				temp.append(init_weight(layer[l-1]))

			row.append(temp)
			temp = []

		weight.append(row)
		row = []

	axisX.append(-1)
	axisY.append(0.5)#in_error(trainX, trainY, weight, layer, threshold))
	axisZ.append(0.5)#out_error(testX, testY, weight, layer, threshold))
	
	# initializing delta matrix
	for i in range(0, num_layers):
		temp.append(1)
		for j in range(1, layer[i]):		
			temp.append(-1)

		delta.append(temp)
		temp = []

	# Stochastic Gradient Descent
	print("Starting stochastic descent")

	inerr = 10000
	outerr = 10000
	itr = 0

	#for itr in range(0, iterations):
	while(inerr>0.02):
		n = rand.randint(0, len(trainX)-1)

		# Forward Propagation
		input_extend = []
		input_extend.append(1)
		input_extend.extend(trainX[n])
		layer_input[0] = np.dot([input_extend], weight[0])[0]
		layer_input[0][0] = 1
		layer_output[0] = (np.exp(np.dot(2, layer_input[0])) - 1)/(1 + np.exp(np.dot(2, layer_input[0])))
		layer_output[0][0] = 1

		for l in range(1, num_layers):
			layer_input[l] = np.dot([layer_output[l-1]], weight[l])[0]
			if(l!=num_layers-1):
				layer_input[l][0] = 1
			
			layer_output[l] = (np.exp(np.dot(2,layer_input[l])) - 1)/(1 + np.exp(np.dot(2, layer_input[l])))
			if(l!=num_layers-1):
				layer_output[l][0] = 1

		# Backward Propagation
		delta[num_layers-1][0] = 2*(layer_output[num_layers-1][0] - trainY[n])*(1 - (layer_output[num_layers-1][0]**2))

		l = num_layers-2
		while(l>=0):
			delta[l] = np.multiply(np.dot([delta[l+1]], np.transpose(weight[l+1]))[0], 1 - np.multiply(layer_output[l], layer_output[l]))
			l = l - 1

		# Update weights
		input_extend = []
		input_extend.append(1)
		input_extend.extend(trainX[n])
		weight[0] = np.subtract(weight[0], np.multiply(alpha, np.dot(np.transpose([input_extend]), [delta[0]])))

		for l in range(1, num_layers):
			weight[l] = np.subtract(weight[l], np.multiply(alpha, np.dot(np.transpose([layer_output[l-1]]), [delta[l]]))) 

		if(itr==0 or itr==10 or itr==20 or itr==30 or itr==50 or itr==100 or itr==200 or itr==300 or itr==500 or itr%1000==0):
			axisX.append(itr)
			inerr = in_error(trainX, trainY, weight, layer, threshold)
			outerr = out_error(testX, testY, weight, layer, threshold)
			axisY.append(inerr)
			axisZ.append(outerr)
			print("	Iteration: "+ str(itr))
			print("		in-sample error = "+ str(inerr) + "  out-of-sample error = "+ str(outerr))
			print("")
			#for a in range(0, len(trainX)):
			#	print(trainY[a], itr, classify(weight, trainX[a], layer, threshold))
		
		itr = itr + 1
		

	fig = plt.figure()
	
	plt.gca().set_color_cycle(['green', 'red'])
	
	plt.plot(axisX, axisY, marker='o')
	plt.plot(axisX, axisZ, marker='o')
	plt.title("Error in tanh")
	plt.xlabel("Iterations")
	plt.ylabel("MSE")
	plt.legend(['In-sample', 'Out-sample'])
	plt.show()
	fig.savefig('tanh.png')
	plt.close(fig)
	print("")
	print("Figure saved as tanh.png")
	print("")

	return weight


def main():
	start = time.time()
	dataset = makeDataset('data.txt', 2000)
	layer = [100, 50, 1]
	weight = learn(dataset, layer, 0.1, 0.5)
	accCalc(dataset[0], dataset[1], dataset[2], dataset[3], weight, layer, 0.5)
	print("")
	print("Time elapsed: "+ str(time.time() - start) +"seconds")

main()
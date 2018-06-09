from random import *
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np

def grad_desc(filename,lamb,alpha,iterations):
	#itr_list = []
	#cost_list = []

	fil = pd.read_csv(filename)
	n = int(float(0.8*fil.shape[0]))

	theta = [[1,1,1,1,1]]

	#for x in range (0,5) :
	#	theta[0].append(randint(1,10))

	theta = np.array(theta)
	temp_theta = theta

	fil['col1'] = 1;

	X = fil[['col1','sqft','floors','bedrooms','bathrooms']].as_matrix()
	Y = fil[['price']].as_matrix()

	X = np.array(X)
	X = X[0:n,:]
	xmean = np.mean(X,0)
	xstd = np.std(X,0)
	X = (X - np.mean(X,0))/np.std(X,0)
	X[:,0] = 1

	Y = np.array(Y)
	Y = Y[0:n,:]
	ymean = np.mean(Y,0)
	ystd = np.std(Y,0)
	Y = (Y - np.mean(Y,0))/np.std(Y,0)

	itr = 0
	while itr < iterations :
		temp_theta = theta*(1-((alpha*lamb)/n)) - (alpha/n)*(np.dot(X.transpose(),(np.dot(X,theta.transpose()) - Y))).transpose()
		
		temp_theta[0,0] = temp_theta[0,0] + theta[0,0]*((alpha*lamb)/n)
		theta = temp_theta

		cost = (np.sum((np.dot(X,theta.transpose()) - Y)**2) + lamb*np.sum(theta**2))/(2*n)
		itr = itr + 1

		#itr_list.append(itr)
		#cost_list.append(cost)
		#print("itr = ",itr," and cost = ",cost)


	for i in range(1,5):
		theta[0,0] = theta[0,0] - (theta[0,i]*xmean[i])/xstd[i]
		theta[0,i] = (theta[0,i]*ystd[0])/xstd[i]
		
	theta[0,0] = (theta[0,0]*ystd[0]) + ymean[0]
	#plt.plot(itr_list,cost_list,marker='o')
	#plt.title("Cost vs iterations")
	#plt.xlabel("Iteration")
	#plt.ylabel("Cost")

	#plt.show()

	return theta


def IRLS(filename,lamb,alpha,iterations):
	#itr_list = []
	#cost_list = []

	fil = pd.read_csv(filename)
	n = int(float(0.8*fil.shape[0]))

	theta = [[1,1,1,1,1]]

	#for x in range (0,5) :
	#	theta[0].append(randint(1,10))

	theta = np.array(theta)
	temp_theta = theta

	fil['col1'] = 1;

	X = fil[['col1','sqft','floors','bedrooms','bathrooms']].as_matrix()
	Y = fil[['price']].as_matrix()

	X = np.array(X)
	X = X[0:n,:]
	xmean = np.mean(X,0)
	xstd = np.std(X,0)
	X = (X - np.mean(X,0))/np.std(X,0)
	X[:,0] = 1

	Y = np.array(Y)
	Y = Y[0:n,:]
	ymean = np.mean(Y,0)
	ystd = np.std(Y,0)
	Y = (Y - np.mean(Y,0))/np.std(Y,0)

	itr = 0
	while itr < iterations :
		temp_theta = theta*(1-((alpha*lamb)/n)) - ((np.dot(np.linalg.inv(np.dot(X.transpose(),X)),np.dot(X.transpose(),(np.dot(X,theta.transpose()) - Y)))).transpose())/n
		
		temp_theta[0,0] = temp_theta[0,0] + theta[0,0]*((alpha*lamb)/n)
		theta = temp_theta

		cost = (np.sum((np.dot(X,theta.transpose()) - Y)**2) + lamb*np.sum(theta**2))/(2*n)
		itr = itr + 1

		#itr_list.append(itr)
		#cost_list.append(cost)
		#print("itr = ",itr," and cost = ",cost)


	for i in range(1,5):
		theta[0,0] = theta[0,0] - (theta[0,i]*xmean[i])/xstd[i]
		theta[0,i] = (theta[0,i]*ystd[0])/xstd[i]
		
	theta[0,0] = (theta[0,0]*ystd[0]) + ymean[0]
	#plt.plot(itr_list,cost_list,marker='o')
	#plt.title("Cost vs iterations")
	#plt.xlabel("Iteration")
	#plt.ylabel("Cost")

	#plt.show()
	return theta


def RMSE(filename,theta):
	fil = pd.read_csv(filename)
	m = fil.shape[0]
	n = int(float(0.8*fil.shape[0]))

	fil['col1'] = 1;

	X = fil[['col1','sqft','floors','bedrooms','bathrooms']].as_matrix()
	Y = fil[['price']].as_matrix()

	X = np.array(X)
	X = X[n:m,:]
	
	#X = (X - np.mean(X,0))/np.std(X,0)
	X[:,0] = 1

	Y = np.array(Y)
	Y = Y[n:m,:]

	#Y = (Y - np.mean(Y,0))/np.std(Y,0)

	rmse_val = np.sum((np.dot(X,theta.transpose())-Y)**2)/(m-n+1)
	rmse_val = math.sqrt(rmse_val)

	return rmse_val


def main():
	lamb = 1000
	alpha = 0.05 
	iterations = 50
	filename = 'data.csv'
	
	fil = open("b_result.txt","w")
	theta = grad_desc(filename,0,alpha,iterations)
	fil.write("Gradient Descent method :-\n")
	fil.write("\tprice = "+str(theta[0,0])+" + "+str(theta[0,1])+"*sqft + "+str(theta[0,2])+"*floors + "+str(theta[0,3])+"*bedrooms + "+str(theta[0,4])+"*bathrooms\n")
	theta = IRLS(filename,0,alpha,iterations)
	fil.write("\nIterative Re-weighted Least Squares method :-\n")
	fil.write("\tprice = "+str(theta[0,0])+" + "+str(theta[0,1])+"*sqft + "+str(theta[0,2])+"*floors + "+str(theta[0,3])+"*bedrooms + "+str(theta[0,4])+"*bathrooms\n")
	
	fil.close()

	itr_list = [10,20,30,40,50,60]
	rmse_grad = []
	rmse_irls = []

	for x in range(0,6):
		rmse_grad.append(RMSE(filename,grad_desc(filename,0,alpha,itr_list[x])))
		rmse_irls.append(RMSE(filename,IRLS(filename,0,alpha,itr_list[x])))

	plt.plot(itr_list,rmse_grad,marker='o')
	plt.title("RMSE vs iterations for gradient descent")
	plt.xlabel("Iterations")
	plt.ylabel("RMSE")
	plt.show()

	plt.plot(itr_list,rmse_irls,marker='o')
	plt.title("RMSE vs iterations for Iterative re-weighted least squares")
	plt.xlabel("Iterations")
	plt.ylabel("RMSE")
	plt.show()


main()


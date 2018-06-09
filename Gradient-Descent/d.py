from random import *
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np

def grad_desc(filename,lamb,alpha,iterations,error):
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
		if error==1 :
			sign = (np.dot(X,theta.transpose())-Y)

			for x in range(0,sign.shape[0]):
				if sign[x,0]<0 :
					sign[x,0] = -1
				else :
					sign[x,0] = 1

			change = (np.dot(X.transpose(),sign)).transpose()

		else:
			if error==2 :
				change = (np.dot(X.transpose(),(np.dot(X,theta.transpose()) - Y))).transpose()

			else:
				change = (np.dot(X.transpose(),(np.dot(X,theta.transpose()) - Y)**2)).transpose()
				iterations = 1
		
		temp_theta = theta*(1-((alpha*lamb)/n)) - (alpha/n)*change
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

	#print(error)
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
	alpha = 0.5 
	iterations = 50
	filename = 'data.csv'
	
	fil = open("d_result.txt","w")

	theta = grad_desc(filename,0,alpha,iterations,1)
	fil.write("Cost function is Mean Absolute Error :-\n")
	fil.write("\tprice = "+str(theta[0,0])+" + "+str(theta[0,1])+"*sqft + "+str(theta[0,2])+"*floors + "+str(theta[0,3])+"*bedrooms + "+str(theta[0,4])+"*bathrooms\n")
	

	theta = grad_desc(filename,0,alpha,iterations,2)
	fil.write("\nCost function is Mean Square Error :-\n")
	fil.write("\tprice = "+str(theta[0,0])+" + "+str(theta[0,1])+"*sqft + "+str(theta[0,2])+"*floors + "+str(theta[0,3])+"*bedrooms + "+str(theta[0,4])+"*bathrooms\n")
	
	theta = grad_desc(filename,0,alpha,iterations,3)
	fil.write("\nCost function is Mean Cube Error :-\n")
	fil.write("\tprice = "+str(theta[0,0])+" + "+str(theta[0,1])+"*sqft + "+str(theta[0,2])+"*floors + "+str(theta[0,3])+"*bedrooms + "+str(theta[0,4])+"*bathrooms\n")
	
	
	fil.close()

	alpha_list = [0.05,0.1,0.2,0.4,0.8,1]
	rmse_absolute = []
	rmse_squared = []
	rmse_cubed = []

	for x in range(0,6):
		rmse_absolute.append(RMSE(filename,grad_desc(filename,0,alpha_list[x],iterations,1)))
		rmse_squared.append(RMSE(filename,grad_desc(filename,0,alpha_list[x],iterations,2)))
		rmse_cubed.append(RMSE(filename,grad_desc(filename,0,alpha_list[x],iterations,3)))

	plt.plot(alpha_list,rmse_absolute,marker='o')
	plt.title("RMSE vs learning rate for mean absolute error")
	plt.xlabel("Learning rate")
	plt.ylabel("RMSE")
	plt.show()

	plt.plot(alpha_list,rmse_squared,marker='o')
	plt.title("RMSE vs learning rate for mean squared error")
	plt.xlabel("Learning rate")
	plt.ylabel("RMSE")
	plt.show()

	plt.plot(alpha_list,rmse_cubed,marker='o')
	plt.title("RMSE vs learning rate for mean cubed error")
	plt.xlabel("Learning rate")
	plt.ylabel("RMSE")
	plt.show()


main()

		
		
		
		
		
		
		
		
	

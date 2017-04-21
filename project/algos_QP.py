import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import fsolve
import time as time


def compute_valueFunction(P_pi,r_pi,discountFactor,
	algo='',nIter_max=0,x_init_=None,alpha=1.0,
	approx=False,features=None,d_stationnary=None,
	verbose=False,history=False,preconditionner=False):
	
	nIter=0
	
	I=np.eye(P_pi.shape[0])
	A=I-discountFactor*P_pi
	b=r_pi
	
	x_history=0.
	
	if(approx):
		
		#compute approximate solution of bellman equation
		
		if( not(x_init_ == None)):
			x_init=np.copy(x_init_)
		else:
			x_init=np.zeros(features.shape[1])
			#x_init=np.random.rand(features.shape[1])
		
		if(algo=='DP' and(features==None or d_stationnary==None)):
			print 'Error calling', algo, ': features and/or stationnary distribution not given'
			return 0, 0., 0.
		
		if(algo=='DP'):
			D=np.diag(d_stationnary)
			b=features.T.dot(D.dot(b))
			A=features.T.dot(D).dot(A).dot(features)
			I=np.eye(features.shape[1])
		else:
			A=A.dot(features)
		
		
	else:
		if( not(x_init_ == None)):
			x_init=np.copy(x_init_)
		else:
			x_init=np.zeros(r_pi.shape)
	
	start_=time.clock()
	
	if(algo=='DP'):
		#dynamic programming
		nIter, x_history=dynamicProgramming(I-alpha*A,alpha*b,x_init,nIter_max,history=history)
		
	elif(algo=='GD'):
		#gradient descent with exact linesearch
		nIter, x_history=gradientDescent(A,b,x_init,nIter_max,history=history)
		
	elif(algo=='GD_am'):
		#gradient descent with adaptive momentum, default parameters
		nIter, x_history=gradientDescent_adaptiveMomentum(A,b,x_init,nIter_max,history=history)
		
	elif(algo=='ADAM'):
		#gradient descent, ADAM
		nIter, x_history=ADAM(A,b,x_init,nIter_max,history=history)
	
	elif(algo=='AdaGrad'):
		#gradient descent, AdaGrad
		nIter, x_history=AdaGrad(A,b,x_init,nIter_max,history=history)
		
	elif(algo=='RMSProp'):
		#gradient descent, RMSProp
		nIter, x_history=RMSProp(A,b,x_init,nIter_max,history=history)
		
	elif(algo=='CG'):
		#conjugate gradient
		nIter, x_history=conjugateGradient(A,b,x_init,nIter_max,history=history)
		
	elif(algo=='Newton'):
		#Newton method (conpute an inverse)
		nIter, x_history=Newton(A,b,x_init,nIter_max=nIter_max,history=history)
		
	end_ = time.clock()
	
	if(verbose):
		print 'Algo:',algo,'\t | Time (s):',end_-start_, '\t | #Iter:', nIter, '\t | T/iter:', (end_-start_)/(nIter)
	
	return nIter,end_-start_,x_history

def dynamicProgramming(P,
					  r,
					  x_init,
					  nIter_max=0,
					  tolerance=10.0**(-14),
						history=False
					  ):
	
	#this function will solve the Bellman Equation x = Px + r
	#using dynamic programming
	if(history):
		X=np.zeros((nIter_max+1,np.size(x_init)))
	x=x_init
	
	keep_going=True
	iter_count=0
	
	for k in range(1,nIter_max+1):
		
		if(keep_going):
			iter_count+=1
			x_new=P.dot(x)+r
			keep_going=np.max(np.abs(x_new-x))>tolerance
			x=x_new
		else:
			break
			
		if(history):
			X[k,:]=x
	
	if(history):
		X[range(iter_count+1,nIter_max+1),:]=x
	else:
		X=x
	return iter_count, X
	
def gradientDescent(A,
					b,
					x_init,
					nIter_max=0,
					tolerance=10.0**(-14),
					history=False
				   ):
	
	#this function will solve Ax=b by performing gradient descent
	#on the following quadratic problem : min ||Ax-b||^2
	
	#A is a matrix, not necessarily symetric
	#b is a vector
	#x_init is a vector
	#nIter_max is the maximum number of iterations of the algorithm
	if(history):
		X=np.zeros((nIter_max+1,np.size(x_init)))
	x=x_init
	
	keep_going=True
	iter_count=0
	
	for k in range(1,nIter_max+1):
		if(keep_going):
			
			iter_count+=1
			#compute current gradient
			c=A.dot(x)-b
			g=A.T.dot(c)
			d=-g

			#compute exact linesearch
			q=A.dot(d)
			t= -q.T.dot(c) / q.T.dot(q)

			#update parameter vector
			x=x+t*d
			
			keep_going=np.max(np.abs(g))>tolerance
			
		if(history):
			X[k,:]=x
	
	if(history):
		X[range(iter_count+1,nIter_max+1),:]=x
	else:
		X=x
	
	return iter_count, X
	
def gradientDescent_adaptiveMomentum(A,
					b,
					x_init,
					nIter_max=0,
					g_momentum=0.9,
					eta=0.1,
					tolerance=10.0**(-14),
					history=False
					):
	
	#this function will solve Ax=b by performing gradient descent
	#on the following quadratic problem : min ||Ax-b||^2
	
	#A is a matrix, not necessarily symetric
	#b is a vector
	#x_init is a vector
	#nIter_max is the maximum number of iterations of the algorithm
	
	if(history):
		X=np.zeros((nIter_max+1,np.size(x_init)))
	
	x=x_init
	momentum=0.0*x
	keep_going=True
	iter_count=0
	
	for k in range(1,nIter_max+1):
		if(keep_going):
			iter_count+=1
			
			#compute current gradient
			g=A.T.dot(A.dot(x)-b)

			#update momentum
			momentum = g_momentum*momentum+ eta*g

			#update parameter vector
			x=x-momentum
			
			keep_going=np.linalg.norm(momentum)>tolerance
		else:
			break
			
		if(history):
			X[k,:]=x
	if(history):
		X[range(iter_count+1,nIter_max+1),:]=x
	else:
		X=x
		
	return iter_count, X
	
def ADAM(A,
		b,
		x_init,
		nIter_max=0,
		eta=0.01,
		beta1=0.9,
		beta2=0.99,
		epsilon=10.0**(-8),
		history=False ):
	
	#this function will solve Ax=b by performing gradient descent
	#on the following quadratic problem : min ||Ax-b||^2
	
	#A is a matrix, not necessarily symetric
	#b is a vector
	#x_init is a vector
	#nIter_max is the maximum number of iterations of the algorithm
	
	if(history):
		X=np.zeros((nIter_max+1,np.size(x_init)))
	
	x=x_init
	m=0.0*x
	v=0.0*x
	
	m_hat=m
	v_hat=v
	
	beta1_k=beta1
	beta2_k=beta2
	
	iter_count=0
	
	for k in range(1,nIter_max+1):
		
		#update number of iterations
		iter_count+=1
		
		#compute current gradient
		g=A.T.dot(A.dot(x)-b)
		
		#update 
		m=beta1*m+(1.-beta1)*g
		v=beta2*v+(1.-beta2)*np.multiply(g,g)
		
		#correct the bias
		m_hat=m/(1.0-beta1_k)
		v_hat=v/(1.0-beta2_k)

		#update parameter vector
		x=x-eta*np.multiply( 1.0 / (np.sqrt(v_hat) + epsilon),m_hat)
		if(history):
			X[k,:]=x
		
		#update beta1_k and beta2_k
		beta1_k *= beta1
		beta2_k *= beta2
	
	if(history):
		X[range(iter_count+1,nIter_max+1),:]=x
	else:
		X=x
	
	return iter_count, X
	
def conjugateGradient(A,
					b,
					x_init=None,
					nIter_max=0,
					epsilon_CG_a=10.0**(-14),
					epsilon_CG_r=10.0**(-14),
					history=False,
					pre_cond_given=False,
					C=None
					):
	
	#This function will solve Ax=b by minimizing the error function |Ax-b|^2
	#When a preconditionner C is given, the function will solve CAx=Cb by minimizing |CAx-Cb|^2
	#The optimization method used is conjugate gradient
	
	if(history):
		X=np.zeros((nIter_max+1,np.size(x_init)))
	
	#initial parameters
	if(pre_cond_given):
		#use preconditionner
		if(x_init==None):
			r=-A.T.dot(C.T.dot(C.dot(b)))
		else:
			r=A.T.dot(C.T.dot(C.dot(A.dot(x_init)-b))) #gradient at the initial point
		
	else:
		#no preconditionner was given
		if(x_init==None):
			r=-A.T.dot(b)
		else:
			r=A.T.dot(A.dot(x_init)-b) #gradient at the initial point
		
		
	p=-r
	r_0_norm=np.linalg.norm(r)
	beta=0.0
	x=x_init
	
	keep_going=True
	iter_count=0
	
	for k in range(1,nIter_max+1):
		
		r_sqnorm=r.dot(r)
		keep_going=(np.sqrt(r_sqnorm) > epsilon_CG_a+epsilon_CG_r*r_0_norm)
		
		#keep updating while stopping criterion is not met
		if(keep_going):
			iter_count+=1
			
			if(not(pre_cond_given)):
				q=A.dot(p) #this speeds up things a bit
			else:
				q=C.dot(A.dot(p))
				
			#compute step size
			alpha = r_sqnorm / q.T.dot(q)

			#perform update on x
			x+=alpha*p

			#compute new conjugate gradient
			if(not(pre_cond_given)):
				r+= alpha*A.T.dot(q)
			else:
				r+= alpha*A.T.dot(C.T.dot(q))
			
			beta= r.dot(r)/ r_sqnorm
			p= - r+beta*p
		else:
			break

		if(history):
			X[k,:]=x
	
	if(history):
		X[range(iter_count+1,nIter_max+1),:]=x
	else:
		X=x
	
	return iter_count, X
	
def AdaGrad(A,
			b,
			x_init,
			nIter_max=0,
			eta=0.1,
			delta=10**(-7),
			epsilon=10.0**(-8),
			history=False ):
	
	if(history):
		X=np.zeros((nIter_max+1,np.size(x_init)))
	
	x=x_init
	r=0.0*x
	g=0.0*x
	
	iter_count=0
	
	for k in range(1,nIter_max+1):
		
		#update number of iterations
		iter_count+=1
		
		#compute current gradient
		g=A.T.dot(A.dot(x)-b)
		
		#accumulate squared gradient
		r+=np.multiply(g,g)
		
		#compute update
		delta_x=-(eta/(delta+np.sqrt(r)))*g
		
		#perform update
		x+=delta_x
	
		if(np.linalg.norm(g)<epsilon):
			break
		
		if(history):
			X[k,:]=x
	
	if(history):
		X[range(iter_count+1,nIter_max+1),:]=x
	else:
		X=x
	
	return iter_count, X
	
def RMSProp(A,
			b,
			x_init,
			nIter_max=0,
			eta=0.01,
			rho=0.9,
			delta=10**(-7),
			epsilon=10.0**(-8),
			history=False ):
	
	if(history):
		X=np.zeros((nIter_max+1,np.size(x_init)))
	
	x=x_init
	r=0.0*x
	g=0.0*x
	
	iter_count=0
	
	for k in range(1,nIter_max+1):
		
		#update number of iterations
		iter_count+=1
		
		#compute current gradient
		g=A.T.dot(A.dot(x)-b)
		
		#accumulate squared gradient
		r=rho*r+(1-rho)*np.multiply(g,g)
		
		#compute update
		delta_x=-(eta/np.sqrt(delta+r))*g
		
		#correct the bias
		x+=delta_x
	
		if(np.linalg.norm(g)<epsilon):
			break
		
		if(history):
			X[k,:]=x
	
	if(history):
		X[range(iter_count+1,nIter_max+1),:]=x
	else:
		X=x
	
	return iter_count, X
	
def Newton(A,
			b,
			x_init,
			nIter_max=0,
			history=False
			):
	#this function will solve Ax=b by computing the inverse of A
	
	#A is a matrix, not necessarily symetric
	#b is a vector
	iter_count=1
	
	if(history):
		X=np.zeros((nIter_max+1,np.size(x_init)))
	
	x=np.linalg.inv(A).dot(b)
	
	if(history):
		X[range(1,nIter_max+1),:]=x
	else:
		X=x
		
	return iter_count, X
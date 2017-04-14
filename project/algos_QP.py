import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import fsolve
import time as time

def dynamicProgramming(P,
                      r,
                      x_init,
                      nIter_max,
                      tolerance=10.0**(-14),
                        history=True
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
                    nIter_max,
                    tolerance=10.0**(-14),
                    history=True
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
            g=A.T.dot(A.dot(x)-b)
            d=-g

            #compute exact linesearch
            q=A.dot(d)
            t= -q.T.dot(A.dot(x)-b) / q.T.dot(q)

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
                    nIter_max,
                    g_momentum=0.9,
                    eta=0.1,
                    tolerance=10.0**(-14),
                    history=True
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
	
def gradientDescent_ADAM(A,
                    b,
                    x_init,
                    nIter_max,
                    eta=0.01,
                    beta1=0.9,
                    beta2=0.99,
                    epsilon=10.0**(-8),
                    history=True ):
    
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
        m=beta1*m+(1-beta1)*g
        v=beta2*v+(1-beta2)*np.multiply(g,g)
        
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
                    x_init,
                    nIter_max,
                    epsilon_CG_a=10.0**(-14),
                    epsilon_CG_r=10.0**(-14),
                    history=True,
                    pre_cond_given=False,
                    C=0.0
                    ):
    
    
    
    #This function will solve Ax=b by minimizing the error function |Ax-b|^2
    #When a preconditionner C is given, the function will solve CAx=Cb by minimizing |CAx-Cb|^2
    #The optimization method used is conjugate gradient
    
    if(history):
        X=np.zeros((nIter_max+1,np.size(x_init)))
    
    #initial parameters
    if(pre_cond_given):
        #use preconditionner
        if(np.linalg.norm(x_init)==0):
            r=-A.T.dot(C.T.dot(C.dot(b)))
        else:
            r=A.T.dot(C.T.dot(C.dot(A.dot(x_init)-b))) #gradient at the initial point
        
    else:
        #no preconditionner was given
        if(np.linalg.norm(x_init)==0):
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
	

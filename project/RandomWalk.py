import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import fsolve
import time as time


import algos_QP

class Random_walk:
	
	def __init__(self, nState, nStep, nFeatures, d_init,gamma,approx=True):
		
		#discount factor
		self.gamma=gamma
		
		#number of states
		self.nState=nState
		self.nStep=nStep
		
		#transition matrix and reward vector
		self.P_pi = np.zeros((nState+1,nState+1))
		self.r_pi=np.zeros(nState+1)
		self.compute_MDP()
		#self.r_pi=0.1*np.random.rand(nState+1)+(1./nState)*np.arange(nState+1)
		
		#initial state distribution
		self.d_init=d_init
		self.d_stationnary=self.d_init
		self.compute_stationnary_distribution()
		
		#features
		self.nFeatures=nFeatures
		self.Phi=np.eye(nState+1)
		if(approx):
			self.create_features() 
		
		#Bellman matrix and vector
		self.P_bellman=self.gamma*self.P_pi
		self.r_bellman=self.r_pi
		
		#A matrix and b vector
		I=np.eye(nState+1)
		self.A_exact=I-self.P_bellman
		self.b_exact=self.r_bellman
		
		if(approx):
			D=np.diag(self.d_stationnary)
			self.A_approx=(self.Phi).T.dot(D).dot(I-self.P_bellman).dot(self.Phi)
			self.b_approx=(self.Phi).T.dot(D).dot(self.r_bellman)
			
		#finally, initialize value function
		self.v_pi=np.zeros(nState+1)
		self.theta=np.zeros(nFeatures+1)
		
		#current state
		self.state=np.random.choice(np.arange(self.nState+1),p=self.d_init)
		
	def restart(self):
		self.state=self.nState/2
		return
	
	def perform_transition(self):
		
		s_now=self.state
		
		#compute new state
		s_new=np.random.choice(np.arange(self.nState+1),p=self.P_pi[s_now,:])
		
		#compute reward
		if(s_new>0):
			r=0
			self.state=s_new
		else:
			if(self.state<self.nStep):
				r=-1
			else:
				r=1
			#if state is terminal, go to initial state, according to initial distribution
			self.state=np.random.choice(np.arange(self.nState+1),p=self.d_init)
		
		return s_now,s_new,self.Phi[s_now,:],self.Phi[s_new,:],r
		
	def compute_MDP(self):
		"""Compute transition matrix and reward vector for the random walk MDP"""

		#Policy is defined as follow:
		#at each time step, go left or right with 50% prob, and take 1 to nStep steps in that direction 
		#(with equal probability)

		#first, we compute the transition matrix P_pi
		#note that for large values of nState, this can be computationnaly and emmory intensive intensive
		#sparse matrices should probably be used here

		rows, cols = np.indices((self.nState+1,self.nState+1))

		for i in range(1,self.nStep+1):
			row_vals_up = np.diag(rows, k=i)
			col_vals_up = np.diag(cols, k=i)
			row_vals_dw = np.diag(rows, k=-i)
			col_vals_dw = np.diag(cols, k=-i)
			self.P_pi[row_vals_up,col_vals_up]+=(1.0/(self.nStep*2.0)) * np.ones(self.nState+1-i)
			self.P_pi[row_vals_dw,col_vals_dw]+=(1.0/(self.nStep*2.0)) * np.ones(self.nState+1-i)

		#state 0 is terminal, so we need to correct the corresponding values in P_pi
		self.P_pi[0,:]=0
		self.P_pi[:,0]=0
		self.P_pi[0,0]=1

		#If the agent hits the boundaries, they reach the terminal state
		#Here we correct the value of P_pi for states are are close to the boundaries
		self.P_pi[np.arange(1,self.nStep+1),0]=0.5-np.arange(0,self.nStep)/(2.0*self.nStep)
		self.P_pi[range(self.nState+1-self.nStep,self.nState+1),0]=0.5-np.arange(self.nStep-1,-1,step=-1)/(2.0*self.nStep)

		#Now, we compute the expected reward vector, r_pi
		#The reward is +1 if reach right end, -1 if reach left end, 0 otherwise
		self.r_pi[range(1,self.nStep+1)]=-(0.5-np.arange(0,self.nStep)/(2.0*self.nStep))
		self.r_pi[range(self.nState+1-self.nStep,self.nState+1)]=0.5-np.arange(self.nStep-1,-1,step=-1)/(2.0*self.nStep)

		return

	def create_features(self):
		"""Compute features matrix"""
		
		#Compute the feature matrix by aggregating states
		self.Phi=np.vstack((np.zeros(self.nFeatures),np.kron(np.eye(self.nFeatures),np.ones((self.nState/self.nFeatures,1)))))
		u=np.zeros(self.nState+1).reshape((self.nState+1,1))
		u[0]=1
		self.Phi=np.hstack((u,self.Phi))

		return 

	def compute_stationnary_distribution(self,nIter=150):
		"""Compute the stationnary distribution of the MDP"""

		
		b=np.zeros(self.nState+2)
		b[self.nState+1]=1.

		#tweak the transition matrix, otherwise all the mass goes to the terminal state
		M=self.P_pi
		M[0,:]=self.d_init

		M=np.eye(self.nState+1)-M.T
		s=np.ones(self.nState+1)
		s[0]=0.0
		M=np.vstack((M,s)) #this extra constraint ensures that d is a probability distribution
							#otherwise, CG returns 0
		
		n,d=algos_QP.conjugateGradient(M,b,nIter_max=nIter,x_init=np.zeros(self.nState+1),
					  epsilon_CG_a=10.**(-14),epsilon_CG_r=10.**(-14),history=False)
		
		self.d_stationnary=d

		return
	
	def compute_valueFunction_exact(self,algo,nIter_max,verbose=False,history=True,preconditionner=False):
		
		v_init=np.zeros(self.nState+1)
		#v_init=np.random.rand(self.nState+1)
		v_history=np.zeros(self.nState+1)
		
		nIter=0
		
		start_=time.time()
		
		if(algo=='DP'):
			#dynamic programming
			nIter, v_history=algos_QP.dynamicProgramming(self.P_bellman,self.r_bellman,v_init,nIter_max,history=history)
			
		elif(algo=='GD'):
			#gradient descent with exact linesearch
			nIter, v_history=algos_QP.gradientDescent(self.A_exact,self.b_exact,v_init,nIter_max,history=history)
			
		elif(algo=='GD_am'):
			#gradient descent with adaptive momentum, default parameters
			nIter, v_history=algos_QP.gradientDescent_adaptiveMomentum(self.A_exact,self.b_exact,v_init,nIter_max,history=history)
			
		elif(algo=='GD_ADAM'):
			#gradient descent, ADAM
			nIter, v_history=algos_QP.gradientDescent_ADAM(self.A_exact,self.b_exact,v_init,nIter_max,history=history)
			
		elif(algo=='CG'):
			#conjugate gradient
			nIter, v_history=algos_QP.conjugateGradient(self.A_exact,
											   self.b_exact,
											   v_init,nIter_max,history=history,
											   pre_cond_given=preconditionner,C=np.eye(self.nState+1)+self.P_bellman)
		
		end_ = time.time()
		
		if(verbose):
			print 'Algo:',algo,'\t | Time (s):',end_-start_, '\t | #Iter:', nIter, '\t | T/iter:', (end_-start_)/nIter
		
		return v_history
	
	def compute_valueFunction_approx(self,algo,nIter_max,verbose=False,alpha=1.0,history=True,preconditionner=False):
		
		theta_init=np.zeros(self.nFeatures+1)
		#theta_init=self.theta+0.0001*np.random.rand(self.nFeatures+1)
		
		
		if(history):
			theta_history=np.zeros(self.nFeatures+1)
		nIter=0
		
		start_=time.time()
		
		if(algo=='DP'):
			#dynamic programming
			nIter, theta_history=algos_QP.dynamicProgramming(np.eye(self.nFeatures+1)-alpha*self.A_approx,
													alpha*self.b_approx,theta_init,nIter_max,history=history)
			
		elif(algo=='GD'):
			#gradient descent with exact linesearch
			nIter, theta_history=algos_QP.gradientDescent(self.A_approx,self.b_approx,theta_init,nIter_max,history=history)
			
		elif(algo=='GD_am'):
			#gradient descent with adaptive momentum, default parameters
			nIter, theta_history=algos_QP.gradientDescent_adaptiveMomentum(self.A_approx,self.b_approx,theta_init,
																  nIter_max,0.8,0.1,history=history)
			
		elif(algo=='GD_ADAM'):
			#gradient descent, ADAM
			nIter, theta_history=algos_QP.gradientDescent_ADAM(self.A_approx,self.b_approx,theta_init,nIter_max,history=history)
			
		elif(algo=='CG'):
			#conjugate gradient
			nIter, theta_history=algos_QP.conjugateGradient(self.A_approx,self.b_approx,theta_init,nIter_max,
												   history=history,pre_cond_given=preconditionner,
												   C=2*np.eye(self.nFeatures+1)-self.A_approx)
		
		end_ = time.time()
		
		if(verbose):
			print 'Algo:',algo,'\t | Time (s):',end_-start_, '\t | #Iter:', nIter, '\t | T/iter:', (end_-start_)/nIter
		
		return theta_history
	
	def run_algo(self,algo,nIter_max,
				 alpha_=0.01,beta_=0.05,gamma_=0.667,
				 zeta_=0.75,eta_=0.1,
				 momentum=False,
				 momentum_type='Regular',
				 epsilon_LSTD=1.0,
				 
				 verbose=False):
		
		start_=time.time()
		
		
		if(algo=='TD'):
			#Classic TD(0)
			theta_history=self.TD(nIter_max,alpha_=alpha_,zeta_=zeta_)
			
		elif(algo=='LSTD'):
			#LSTD
			theta_history=self.LSTD(nIter_max,epsilon=epsilon_LSTD)
		
		elif(algo=='LLSTD'):
			#Limited-memory LSTD
			theta_history=self.LLSTD(nIter_max,epsilon=epsilon_LSTD)
			
		elif(algo=='GTD'):
			#Gradient-TD
			theta_history=self.GTD(nIter_max,alpha_=alpha_,beta_=beta_,gamma_=gamma_,
								   zeta_=zeta_,eta_=eta_,
								   momentum=momentum,
								  momentum_type=momentum_type)
			
		elif(algo=='GTD2'):
			#gradient-TD 2 
			theta_history=self.GTD2(nIter_max,alpha_=alpha_,beta_=beta_,gamma_=gamma_,
								   zeta_=zeta_,eta_=eta_,
								   momentum=momentum,
								  momentum_type=momentum_type)
		
			
		elif(algo=='TDC'):
			#TD with gradient correction
			theta_history=self.TDC(nIter_max,alpha_=alpha_,beta_=beta_,gamma_=gamma_,
								   zeta_=zeta_,eta_=eta_,
								   momentum=momentum,
								  momentum_type=momentum_type)
			
		else:
			print 'Error :',algo,'is an unknown algorithm'
			return np.zeros((nIter_max+1,self.nFeatures+1))
		
		end_ = time.time()
		
		if(verbose):
			print 'Algo:',algo,'\t | Time (s):',end_-start_
		
		return theta_history
	
	def TD(self,nIter_max,alpha_=0.01,zeta_=0.75):
		
		theta=np.zeros(self.nFeatures+1)
		theta_history=np.zeros((nIter_max+1,self.nFeatures+1))
		delta_history=np.zeros((nIter_max+1,self.nFeatures+1))
 
		for k in range(1,nIter_max+1):
			
			#perform transition	
			s_now,s_new,phi_now,phi_new,reward=self.perform_transition()
			delta = reward+self.gamma*theta.dot(phi_new)-theta.dot(phi_now)

			alpha=alpha_/(1.+0.01*k**(zeta_))
			
			#update parameter
			theta += alpha*delta*phi_now
			#print k, delta
			
			
			theta_history[k,:]=theta
			delta_history[k,:]=delta*phi_now

		return theta_history
	
	def GTD(self,nIter_max,
			alpha_=0.1,beta_=0.5,gamma_=0.9,
			zeta_=0.75,eta_=0.01,
			momentum=False,momentum_type='Regular'):
		
		theta=np.zeros(self.nFeatures+1)
		theta_history=np.zeros((nIter_max+1,self.nFeatures+1))
		
		w=np.zeros(self.nFeatures+1)
		alpha=alpha_
		beta=beta_
		
		cos_phi=np.zeros(nIter_max+1)
		
		for k in range(1,nIter_max+1):
			
			#perform transition	
			s_now,s_new,phi_now,phi_new,reward=self.perform_transition()
			delta = reward+self.gamma*theta.dot(phi_new)-theta.dot(phi_now)
			
			#update parameter
			g=-(phi_now.T.dot(w))*(phi_now-self.gamma*phi_new)
			delta_theta = alpha*(phi_now.T.dot(w))*(phi_now-self.gamma*phi_new)
			theta += delta_theta
			w+=beta*(delta* phi_now - w)
			
			theta_history[k,:]=theta
		
		return theta_history
	
	def GTD2(self,nIter_max,
			alpha_=0.1,beta_=0.5,gamma_=0.9,
			zeta_=0.75,eta_=0.01,
			momentum=False,momentum_type='Regular'):
		
		theta=np.zeros(self.nFeatures+1)
		theta_history=np.zeros((nIter_max+1,self.nFeatures+1))
		
		w=np.zeros(self.nFeatures+1)
		alpha=alpha_
		beta=beta_
			  
		for k in range(1,nIter_max+1):
			
			#perform transition	
			s_now,s_new,phi_now,phi_new,reward=self.perform_transition()
			delta = reward+self.gamma*theta.dot(phi_new)-theta.dot(phi_now)
			
			#update parameter
			theta += alpha*(phi_now.T.dot(w))*(phi_now-self.gamma*phi_new)
			w+=beta*(delta-phi_now.T.dot(w))*phi_now

			theta_history[k,:]=theta
		
		return theta_history
	
	def GTD2_momentum(self,nIter_max,alpha=0.1,beta=0.1,gamma_m=0.9):
		
		theta=np.zeros(self.nFeatures+1)
		theta_history=np.zeros((nIter_max+1,self.nFeatures+1))
		
		w=np.zeros(self.nFeatures+1)
		v=np.zeros(self.nFeatures+1)
		alpha_=alpha
			  
		for k in range(1,nIter_max+1):
			
			#perform transition	
			s_now,s_new,phi_now,phi_new,reward=self.perform_transition()
			delta = reward+self.gamma*theta.dot(phi_new)-theta.dot(phi_now)
			
			#update parameter
			alpha=alpha_/(1.0+0.001*k)
			g = -(phi_now.T.dot(w))*(phi_now-self.gamma*phi_new)
			v=gamma_m*v+alpha*g
			
			theta += -v
			w+=beta*(delta-phi_now.T.dot(w))*phi_now

			theta_history[k,:]=theta
		
		return theta_history
	
	def TDC(self,nIter_max,
			alpha_=0.1,beta_=0.5,gamma_=0.9,
			zeta_=0.75,eta_=0.01,
			momentum=False,momentum_type='Regular'):
		
		theta=np.zeros(self.nFeatures+1)
		theta_history=np.zeros((nIter_max+1,self.nFeatures+1))
		
		w=np.zeros(self.nFeatures+1)
		v=np.zeros(self.nFeatures+1)
		beta=beta_
		alpha=alpha_
			  
		for k in range(1,nIter_max+1):
			
			#perform transition	
			s_now,s_new,phi_now,phi_new,reward=self.perform_transition()
			delta = reward+self.gamma*theta.dot(phi_new)-theta.dot(phi_now)
			
			#compute gradient estimate
			g=-delta*phi_now +self.gamma*(phi_now.T.dot(w))*phi_new
			
			#Compute parameters updates
			#beta=alpha_/(k**(zeta_+eta_))
			delta_w = beta*(delta-phi_now.dot(w))*phi_now
			
			if(not(momentum)):
				#No momentum
				#alpha=alpha_/(1.+0.01*k**(zeta_))
				delta_theta=alpha*delta*phi_now -alpha*self.gamma*(phi_now.T.dot(w))*phi_new
				
			elif(momentum_type=='Nesterov'):
				#Nesterov Momentum
				
				#first, compute gradient at interim point
				delta_nesterov=reward+self.gamma*(theta-gamma_*v).dot(phi_new)-(theta-gamma_*v).dot(phi_now)
				g_nesterov=-delta_nesterov*phi_now +self.gamma*(phi_now.T.dot(w))*phi_new
				#update velocity
				v=gamma_*v-alpha_*g_nesterov
				delta_theta=v
				
			else:
				#Regular momentum
				v=gamma_*v-alpha_*g
				delta_theta=v
				
			
			#Update theta and w
			theta += delta_theta
			w+=delta_w
			
			theta_history[k,:]=theta
		
		return theta_history
	
	def LSTD(self,
			nIter_max,
			epsilon
			):
		
		theta=np.zeros(self.nFeatures+1)
		theta_history=np.zeros((nIter_max+1,self.nFeatures+1))
		
		A_inv=(1.0/epsilon)*np.eye(self.nFeatures+1)
		b_=np.zeros(self.nFeatures+1)
		
		for k in range(1,nIter_max+1):
			s_now,s_new,phi_now,phi_new,reward=self.perform_transition()
			#print k, s_now, s_new, reward
			
			v=A_inv.T.dot(phi_now-self.gamma*phi_new)
			
			A_inv=A_inv - np.outer(A_inv.dot(phi_now),v) / (1.0+v.T.dot(phi_now))
			b_+=reward*phi_now
			
			
			theta=A_inv.dot(b_)
			#print k, theta
			
			theta_history[k,:]=theta
		
		return theta_history
	
	def LLSTD(self,
			 nIter_max,
			 alpha=0.1,
			 epsilon=1.,
			 m=1000
			 ): 
		
		m=int(np.sqrt(self.nFeatures+1))
		#m=nIter_max
		#m=50
		
		theta=np.zeros(self.nFeatures+1)
		theta_history=np.zeros((nIter_max+1,self.nFeatures+1))
		
		batch_phi_now=np.zeros((m,self.nFeatures+1))
		batch_phi_new=np.zeros((m,self.nFeatures+1))
		batch_reward=np.zeros(m)
		
		memory_b=np.zeros((m+1,self.nFeatures+1))
		memory_B_phi=np.zeros((m+1,m+1,self.nFeatures+1))
		
		b_bis=np.zeros(self.nFeatures+1)
		d_history=np.nan*np.ones((nIter_max+1,self.nFeatures+1))
	
		count=0.
		delta=0.1
		
		
		
		for k in range(1,nIter_max+1):
			
			s_now,s_new,phi_now,phi_new,reward=self.perform_transition()
			#print k, s_now, s_new, reward
			
			b_bis+=(1./k)*(reward*phi_now-b_bis)
			
			#update the batch
			batch_phi_now[k%(m),:]=phi_now
			batch_phi_new[k%(m),:]=phi_new
			batch_reward[k%(m)]=reward
			
			
			
			if(k%(m)==0 and (k>=m)):
				
				memory_b=np.zeros((m+1,self.nFeatures+1))
				memory_B_phi=np.zeros((m+1,m+1,self.nFeatures+1))
				b_=np.zeros(self.nFeatures+1)
				count+=1.
				
				#Initialize B_0 * Phi_l, forall l
				# memory_B_phi[t,l,:] is B_t * Phi_l
				for l in range(m):
					phi_now=batch_phi_now[(k-m+1+l)%m]
					memory_B_phi[0,l+1,:]=(1./epsilon)*phi_now
					b_+=batch_reward[(k-m+1+l)%m]*phi_now
				
				#memory_b[0,:]=(1./epsilon)*b_
				memory_b[0,:]=(m/(epsilon))*b_bis
				
				#now, iteratively compute A^{-1}*b
				for t in range(1,m+1):
					
					#Here, we compute all products B_t * Phi_u
					u=batch_phi_now[(k-m+t)%m]-self.gamma*batch_phi_new[(k-m+t)%m]
					w=memory_B_phi[t-1,t,:]
					for l in range(t,m+1):
						
						v=memory_B_phi[t-1,l,:]
						memory_B_phi[t,l,:]=v-((u.dot(v))/(1.+u.dot(w)))*(w)
					
					#Now, update B_t*b_
					v=memory_b[t-1,:]
					
					#Compute B_t * b
					memory_b[t,:]=v - ((u.dot(v))/(1.+u.dot(w)))*(w)
					
					#print t, memory_b[t,:]
					
					
				#Restrict the magnitude of updates (similar to using trust-region)
				d=memory_b[m,:]
				
				#if(np.linalg.norm(d)>delta):
					#d=delta*d/np.linalg.norm(d)
				
				d_history[k]=d
				theta+=(1./count)*(d-theta)
				
				
			
			theta_history[k,:]=theta

		return theta_history
	

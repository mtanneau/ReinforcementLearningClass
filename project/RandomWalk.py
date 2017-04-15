import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import fsolve
import time as time


import algos_QP

class Random_walk:
	
	def __init__(self,gamma, nState, nAction, behaviour_policy, d_init, feature_matrix=None):
		
		#discount factor
		self.gamma=gamma
		
		#state and action space
		self.nState=nState
		self.nAction=nAction
		
		#behaviour policy
		self.policy=behaviour_policy
		
		#feature matrix
		self.Phi=feature_matrix
		
		#transition matrix and reward vector
		#self.P_pi = np.zeros((nState+1,nState+1))
		#self.r_pi=np.zeros(nState+1)
		#self.compute_MDP()
		
		#initial state distribution
		self.d_init=d_init
		#self.d_stationnary=self.d_init
		#self.compute_stationnary_distribution()
		
		#current state
		self.current_state=np.random.choice(np.arange(self.nState+1),p=self.d_init)
		
		#features
		if(not feature_matrix==None)
			self.nFeatures=feature_matrix.shape[1]-1

	@staticmethod
	def compute_policy(nState, nStep):
		
		#Policy is defined as follow:
		#at each time step, go left or right with 50% prob, and take 1 to nStep steps in that direction 
		#(with equal probability)

		#first, we compute the transition matrix P_pi
		#note that for large values of nState, this can be computationnaly and emmory intensive intensive
		#sparse matrices should probably be used here
		
		P_pi=np.zeros((nState+1,nState+1))
		r_pi=np.zeros(nState+1)

		rows, cols = np.indices((nState+1,nState+1))

		for i in range(1,nStep+1):
			row_vals_up = np.diag(rows, k=i)
			col_vals_up = np.diag(cols, k=i)
			row_vals_dw = np.diag(rows, k=-i)
			col_vals_dw = np.diag(cols, k=-i)
			P_pi[row_vals_up,col_vals_up]+=(1.0/(nStep*2.0)) * np.ones(nState+1-i)
			P_pi[row_vals_dw,col_vals_dw]+=(1.0/(nStep*2.0)) * np.ones(nState+1-i)

		#state 0 is terminal, so we need to correct the corresponding values in P_pi
		P_pi[0,:]=0.
		P_pi[:,0]=0.
		P_pi[0,0]=1.

		#If the agent hits the boundaries, they reach the terminal state
		#Here we correct the value of P_pi for states are are close to the boundaries
		P_pi[np.arange(1,nStep+1),0]=0.5-np.arange(0,nStep)/(2.0*nStep)
		P_pi[range(nState+1-nStep,nState+1),0]=0.5-np.arange(nStep-1,-1,step=-1)/(2.0*nStep)

		#Now, we compute the expected reward vector, r_pi
		#The reward is +1 if reach right end, -1 if reach left end, 0 otherwise
		r_pi[range(1,nStep+1)]=-(0.5-np.arange(0,nStep)/(2.0*nStep))
		r_pi[range(nState+1-nStep,nState+1)]=0.5-np.arange(nStep-1,-1,step=-1)/(2.0*nStep)
		
		return P_pi,r_pi
	
	def restart(self):
		self.current_state=np.random.choice(np.arange(self.nState+1),p=self.d_init)
		return
	
	def perform_transition(self):
		
		s_now=self.current_state
		
		#select action
		a=np.random.choice(np.arange(self.nAction+1),p=self.policy[s_now,:])
		
		#probability that this action be taken under behaviour policy
		mu=self.policy(s_now,a) 
		
		#compute new state
		#here, the action gives the new state directly
		s_new=a
		
		#compute reward
		if(s_new>0):
			r=0
			self.current_state=s_new
		else:
			if(self.state<self.nStep):
				r=-1
			else:
				r=1
			#if state is terminal, go to initial state, according to initial distribution
			self.restart()
		
		return s_now,s_new,self.Phi[s_now,:],self.Phi[s_new,:],r, mu

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
	
	def run_algo(self,algo,nIter_max,
				 alpha_=0.01,beta_=0.05,gamma_=0.667,
				 zeta_=0.75,eta_=0.1,
				 momentum=False,
				 momentum_type='Regular',
				 epsilon_LSTD=1.0,
				 
				 verbose=False):
		
		start_=time.clock()
		
		
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
								   momentum=momentum)
			
		elif(algo=='GTD2'):
			#gradient-TD 2 
			theta_history=self.GTD2(nIter_max,alpha_=alpha_,beta_=beta_,gamma_=gamma_,
								   zeta_=zeta_,eta_=eta_,
								   momentum=momentum)
		
		elif(algo=='GTD3'):
			#gradient-TD 2 
			theta_history=self.GTD3(nIter_max,alpha_=alpha_,gamma_=gamma_,
								   momentum=momentum)
		
			
		elif(algo=='TDC'):
			#TD with gradient correction
			theta_history=self.TDC(nIter_max,alpha_=alpha_,beta_=beta_,gamma_=gamma_,
								   zeta_=zeta_,eta_=eta_,
								   momentum=momentum)
			
		elif(algo=='TDC_AdaGrad'):
			#TD with gradient correction and AdaGrad adaptive learning rate
			theta_history=self.TDC_AdaGrad(nIter_max,alpha_=alpha_,beta_=beta_,gamma_=gamma_,
								   zeta_=zeta_,eta_=eta_,
								   momentum=momentum)
		
		elif(algo=='TDC_ADAM'):
			#TD with gradient correction
			theta_history=self.TDC_ADAM(nIter_max,alpha_=alpha_,beta_=beta_,gamma_=gamma_,
								   zeta_=zeta_,eta_=eta_,
								   momentum=momentum)
			
		else:
			print 'Error :',algo,'is an unknown algorithm'
			return np.zeros((nIter_max+1,self.nFeatures+1))
		
		end_ = time.clock()
		
		if(verbose):
			print 'Algo:',algo,'\t | Time (s):',end_-start_
		
		return theta_history
	
	def TD(self,nIter_max,alpha_=0.01,zeta_=0.75):
		
		theta=np.zeros(self.nFeatures+1)
		theta_history=np.zeros((nIter_max+1,self.nFeatures+1))
		delta_history=np.zeros((nIter_max+1,self.nFeatures+1))
 
		for k in range(1,nIter_max+1):
			
			#perform transition	
			s_now,s_new,phi_now,phi_new,reward, rho =self.perform_transition()
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
			s_now,s_new,phi_now,phi_new,reward, rho=self.perform_transition()
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
			momentum_type='None'):
		
		theta=np.zeros(self.nFeatures+1)
		theta_history=np.zeros((nIter_max+1,self.nFeatures+1))
		
		w=np.zeros(self.nFeatures+1)
		alpha=alpha_
		beta=beta_
			  
		for k in range(1,nIter_max+1):
			
			#perform transition	
			s_now,s_new,phi_now,phi_new,reward, rho=self.perform_transition()
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
			s_now,s_new,phi_now,phi_new,reward,rho=self.perform_transition()
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
			momentum='None'):
		
		theta=np.zeros(self.nFeatures+1)
		theta_history=np.zeros((nIter_max+1,self.nFeatures+1))
		
		w=np.zeros(self.nFeatures+1)
		v=np.zeros(self.nFeatures+1)
		beta=beta_
		alpha=alpha_
			  
		for k in range(1,nIter_max+1):
			
			#perform transition	
			s_now,s_new,phi_now,phi_new,reward,rho=self.perform_transition()
			delta = reward+self.gamma*theta.dot(phi_new)-theta.dot(phi_now)
			
			#compute gradient estimate
			g=-delta*phi_now +self.gamma*(phi_now.T.dot(w))*phi_new
			
			#Compute parameters updates
			#beta=alpha_/(k**(zeta_+eta_))
			delta_w = beta*(delta-phi_now.dot(w))*phi_now
			
			if(momentum=='None'):
				#No momentum
				#alpha=alpha_/(1.+0.01*k**(zeta_))
				delta_theta=alpha*delta*phi_now -alpha*self.gamma*(phi_now.T.dot(w))*phi_new
				
			elif(momentum=='Nesterov'):
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
			s_now,s_new,phi_now,phi_new,reward,rho=self.perform_transition()
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
		
		m=int(np.sqrt(self.nFeatures+1))/2
		#m=nIter_max
		#m=100
		#m=10
		theta_true=np.linalg.inv(self.A_approx).dot(self.b_approx)
		
		rnd_gen=np.random.RandomState()
		rnd_gen.seed(0)
		
		batch_size=20*m
		
		theta=np.zeros(self.nFeatures+1)
		theta_history=np.zeros((nIter_max+1,self.nFeatures+1))
		
		batch=[() for i in range(batch_size)]
		
		#batch_phi_now=np.zeros((m,self.nFeatures+1))
		#batch_phi_new=np.zeros((m,self.nFeatures+1))
		#batch_reward=np.zeros(m)
		
		memory_b=np.zeros((m+1,self.nFeatures+1))
		memory_B_phi=np.zeros((m+1,m+1,self.nFeatures+1))
		
		b_bis=np.zeros(self.nFeatures+1)
		d_history=np.nan*np.ones((nIter_max+1,self.nFeatures+1))
		theta_target_history=np.nan*np.ones((nIter_max+1,self.nFeatures+1))
	
		count=0.
		delta=0.1
		
		for k in range(1,batch_size+1):
			s_now,s_new,phi_now,phi_new,reward,rho=self.perform_transition()
			batch[k-1]=(phi_now,phi_new,reward)
			b_bis+=(1./k)*(reward*phi_now-b_bis)
		
		for k in range(batch_size,nIter_max+1):
			
			#observe transition
			s_now,s_new,phi_now,phi_new,reward,rho=self.perform_transition()
			
			#update b
			b_bis+=(1./k)*(reward*phi_now-b_bis)
					
				
			index=rnd_gen.randint(batch_size,size=m+1)
			
			#update the batch
			#the new transition replaces a random old one
			batch[index[m]]=(phi_now,phi_new,reward)
			#print k, index
			
			if( (k%(m)==0 and (k>=m)) or True):
				
				memory_b=np.zeros((m+1,self.nFeatures+1))
				memory_B_phi=np.zeros((m+1,m+1,self.nFeatures+1))
				b_=np.zeros(self.nFeatures+1)
				count+=1.
				
				#Initialize B_0 * Phi_l, forall l
				# memory_B_phi[t,l,:] is B_t * Phi_l
				A=np.zeros((self.nFeatures+1,self.nFeatures+1))
				for l in range(m):
					transition=batch[index[l]]
					phi_now=transition[0]
					phi_new=transition[1]
					reward=transition[2]
					memory_B_phi[0,l+1,:]=(1./epsilon)*phi_now
					b_+=reward*phi_now
					A+=np.outer(phi_now, phi_now-self.gamma*phi_new)
					
				#print count, b_
				memory_b[0,:]=(1./epsilon)*b_
				#memory_b[0,:]=(m/(epsilon))*b_bis
				
				#now, iteratively compute A^{-1}*b
				for t in range(1,m+1):
					
					transition=batch[index[t-1]]
					phi_now=transition[0]
					phi_new=transition[1]
					#Here, we compute all products B_t * Phi_u
					
					u=phi_now-self.gamma*phi_new
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
				theta_target=memory_b[m,:]
				theta_target_history[k]=theta_target
				#d=memory_b[m,:]-theta
				#d_=np.linalg.inv(epsilon*np.eye(self.nFeatures+1)+A).dot(m*b_bis)
				#d_true=-theta+theta_true
				#print k, theta_target
				
				#update step size
				#d_before=d_history[k-1]
				#cos_=d.dot(d_before)/(np.linalg.norm(d)*(np.linalg.norm(d_before)))
				#print count, cos_
				#if(cos_<0):
					#print count, 'Reducing trust radius', cos_
					#delta=0.9*delta
				
				#compute step size
				#d=delta*d/np.linalg.norm(d)
				
				#print count, d.dot(d_true)/(np.linalg.norm(d)*np.linalg.norm(d_true)), cos_
				
				#d_history[k]=d
				theta+=(1./count)*(theta_target-theta)
				
				
			
			theta_history[k,:]=theta
			
		#print 'LLSTD-count',count
		
		#print np.nanmean(theta_target_history,0)
		#print np.nanstd(theta_history,0)
		
		# plt.plot(b_bis,'b-')
		# plt.plot(self.b_approx,'r--')
		plt.plot(np.nanmean(theta_target_history,0))
		plt.plot(theta_true,'r--')
		plt.show()
		
		return theta_history
	
	def TDC_AdaGrad(self,nIter_max,
			alpha_=0.1,beta_=0.5,gamma_=0.9,
			zeta_=0.75,eta_=0.01,
			momentum='None'):
		
		theta=np.zeros(self.nFeatures+1)
		r=np.zeros(self.nFeatures+1)
		
		theta_history=np.zeros((nIter_max+1,self.nFeatures+1))
		
		w=np.zeros(self.nFeatures+1)
		
		beta=beta_
		alpha=alpha_
			  
		for k in range(1,nIter_max+1):
			
			#perform transition	
			s_now,s_new,phi_now,phi_new,reward,rho=self.perform_transition()
			delta = reward+self.gamma*theta.dot(phi_new)-theta.dot(phi_now)
			
			#compute gradient estimate
			g=-delta*phi_now +self.gamma*(phi_now.T.dot(w))*phi_new
			
			#accumulate squarred gradients
			r+=np.multiply(g,g)
			
			#Compute updates
			delta_theta = -(alpha_/(10**(-7)+np.sqrt(r)))*g
			delta_w = beta*(delta-phi_now.dot(w))*phi_now
			
			#Update theta and w
			theta += delta_theta
			w+=delta_w
			
			theta_history[k,:]=theta
		
		return theta_history
		
	def TDC_ADAM(self,nIter_max,
			alpha_=0.1,beta_=0.5,gamma_=0.9,
			zeta_=0.75,eta_=0.01,
			rho1=0.9,rho2=0.999,
			momentum='None'):
		
		theta=np.zeros(self.nFeatures+1)
		r=np.zeros(self.nFeatures+1)
		s=np.zeros(self.nFeatures+1)
		
		theta_history=np.zeros((nIter_max+1,self.nFeatures+1))
		
		w=np.zeros(self.nFeatures+1)
		
		beta=beta_
		alpha=alpha_
		
		rho1_t=rho1
		rho2_t=rho2
			  
		for k in range(1,nIter_max+1):
			
			#perform transition	
			s_now,s_new,phi_now,phi_new,reward,rho=self.perform_transition()
			delta = reward+self.gamma*theta.dot(phi_new)-theta.dot(phi_now)
			
			#compute gradient estimate
			g=-delta*phi_now +self.gamma*(phi_now.T.dot(w))*phi_new
			
			#update moments estimates
			s = rho1*s + (1.-rho1)*g
			r=rho2*r+(1.-rho2)*np.multiply(g,g)
			
			#Correct bias
			s=s/(1.-rho1_t)
			r=r/(1.-rho2_t)
			
			rho1_t*=rho1
			rho2_t*=rho2
			
			#Compute updates
			delta_theta = -alpha_*(s/(10**(-8)+np.sqrt(r)))
			delta_w = beta*(delta-phi_now.dot(w))*phi_now
			
			#Update theta and w
			theta += delta_theta
			w+=delta_w
			
			theta_history[k,:]=theta
		
		return theta_history
import numpy as np
from itertools import permutations

def pdf(x,bins):
	H, edges = np.histogramdd(x,bins)
	return H/np.sum(H)

# def uniquelist(x):

# 	vals, inds = np.unique(x, return_index=True)
# 	x1=np.copy(x)
# 	binsx=len(vals)
	
# 	for i in range(len(x)):
# 		x1[i]= np.where(vals==x[i])[0][0]
# 	return x1,binsx

def uniquelist(x):

	vals, x1 = np.unique(x, return_inverse=True)
	binsx=len(vals)
	return x1,binsx

def Entropy(x):

	xu,binsx=uniquelist(x)

	Px = pdf(xu,binsx)
	
	E=0.0
	for i in range(binsx):
		if Px[i]>0:
			E+=-Px[i]*np.log(Px[i])
	return E

def I(x,y):

	xu,binsx=uniquelist(x)
	yu,binsy=uniquelist(y)

	Px = pdf(xu,binsx)
	Py = pdf(yu,binsy)
	Pxy = pdf([xu,yu],(binsx,binsy))
	
	MI=0.0
	for i in range(binsx):
		for j in range(binsy):
			if Pxy[i,j]>0:
				MI+=Pxy[i,j]*np.log(Pxy[i,j]/(Px[i]*Py[j]))
	return MI


def Imin(x,y,z):

	xu,binsx=uniquelist(x)
	yu,binsy=uniquelist(y)
	zu,binsz=uniquelist(z)
	
	Px = pdf(xu,binsx)
	Py = pdf(yu,binsy)
	Pz = pdf(zu,binsz)
	Pxy = pdf([xu,yu],(binsx,binsy))
	Pxz = pdf([xu,zu],(binsx,binsz))
	
	I=0.0
	for i in range(binsx):
		Ixy=0
		Ixz=0
		for j in range(binsy):
			if Pxy[i,j]>0:
				Ixy+=Pxy[i,j]*np.log(Pxy[i,j]/(Px[i]*Py[j]))
		for j in range(binsz):
			if Pxz[i,j]>0:
				Ixz+=Pxz[i,j]*np.log(Pxz[i,j]/(Px[i]*Pz[j]))
		I+=min(Ixy,Ixz)
				
	return I

def ConditionalEntropy(x,y):

	xu,binsx=uniquelist(x)
	yu,binsy=uniquelist(y)

	Py = pdf(yu,binsy)
	Pxy = pdf([xu,yu],(binsx,binsy))
	
	
	CE=0.0
	for i in range(binsx):
		for j in range(binsy):
			if Pxy[i,j]>0:
				CE+=Pxy[i,j]*np.log(Py[j]/(Pxy[i,j]))


"""
Transfer entropy of discretized time series x and y, computing transfer entropy form x to a future state of y
x = source series
y = sink series
r = distance of future state
d = number of samples of present state
l = distance between samples of present state
"""
def TE(x,y,r=1,d=1,l=1):

	xu,_=uniquelist(x)
	yu,binsy2=uniquelist(y)
	nx=len(np.unique(xu))
	ny=len(np.unique(yu))
	L=min([len(x),len(y)])-r-(d-1)*l
	
	x1=xu[(d-1)*l:L+(d-1)*l]
	y1=yu[(d-1)*l:L+(d-1)*l]
	for i in range(1,d):
		x1=nx*x1 + xu[(d-1)*l-i:L+(d-1)*l-i]
		y1=ny*y1 + yu[(d-1)*l-i:L+(d-1)*l-i]
	y2=yu[r+(d-1)*l:L+r+(d-1)*l]
	
	x1,binsx1=uniquelist(x1)
	y1,binsy1=uniquelist(y1)

	Px1y1y2 = pdf([x1,y1,y2],(binsx1,binsy1,binsy2))
	Py1 = pdf(y1,binsy1)
	Py1y2 = pdf([y1,y2],(binsy1,binsy2))
	Px1y1 = pdf([x1,y1],(binsx1,binsy1))
	
	TE=0.0
	for i in range(binsx1):
		for j in range(binsy1):
			for k in range(binsy2):
				if Px1y1y2[i,j,k]>0:
					TE+=Px1y1y2[i,j,k]*np.log(Px1y1y2[i,j,k]*Py1[j]/(Py1y2[j,k]*Px1y1[i,j]))
	return TE
	
	

	
def nhist(x,bins):
	
	if len(x.shape)==1:
		M=1
		N=len(x)
	else:
		(M,N)=x.shape
	I=np.zeros(shape=(bins,) * M)
	S=I.shape

	Sub=()
	if M==1:
		idx=x
		I=1.0/N*np.bincount(idx,minlength=bins).astype(float)
	else:
		for i in range(M):
			Sub=Sub+(x[i,:],)
		idx= np.ravel_multi_index(Sub, dims=S, order='F')
		I1=1.0/N*np.bincount(idx,minlength=bins**M).astype(float)
		I=np.reshape(I1, S, order='F')

	return I


def permdist(x,m):
	L=len(x)
	P=np.array(list(permutations(range(m))))
	NP=P.shape[0]
	
	x=x + np.linspace(0, 1E-10, num=L)
	P1=np.zeros((m,L-m+1)).astype(int)
	for i in range(m):
		for j in range(m):
			P1[j,:]=P1[j,:]+(x[j:L-m+j+1]>x[i:L-m+i+1]).astype(int)
	P1=np.transpose(P1)
	
	y=np.zeros(L-m+1).astype(int)
	for i in range(NP):
		idx=np.where(np.prod(P1==np.tile(P[i,:],(L-m+1,1)), axis=1))
		y[idx[0]]=i

	return y
	
"""
Symbolic transfer entropy of discretized time series x and y, computing transfer entropy form x to a future state of y
using symbolization based on permutation
x = source series
y = sink series
m = number of bits per symbol
r = distance of future state
"""
def SymbolicTransferEntropy(x,y,m,r):

	L=min([len(x),len(y)])-m-r+1
	x1=permdist(x[0:L],m);
	y1=permdist(y[0:L],m);
	y2=permdist(y[m+r-1:L+m+r-1],m);
	
	bins=np.math.factorial(m)
	Px1y1y2=nhist(np.vstack([x1,y1,y2]),bins);
	Py1=nhist(y1,bins);
	Py1y2=nhist(np.vstack([y1,y2]),bins);
	Px1y1=nhist(np.vstack([x1,y1]),bins);
	
	TE=0.0
	for i in range(bins):
		for j in range(bins):
			for k in range(bins):
				if Px1y1y2[i,j,k]>0:
					TE+=Px1y1y2[i,j,k]*np.log(Px1y1y2[i,j,k]*Py1[j]/(Py1y2[j,k]*Px1y1[i,j]))
	return TE/np.log(m)


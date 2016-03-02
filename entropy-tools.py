import numpy as np

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

from itertools import permutations

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
	
def TransferEntropy(x,y,m,r):

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

def pdf(x,bins):
	H, edges = np.histogramdd(x,bins)
	return H/np.sum(H)
	
def uniquelist(x):

	vals, inds = np.unique(x, return_index=True)
	x1=np.copy(x)
	
	for i in range(len(x)):
		x1[i]= np.where(vals==x[i])[0][0]
	return x1
def Entropy(x):

	xu=uniquelist(x)
	binsx=int(max(x)-min(x)+1)

	Px = pdf(xu,binsx)
	
	E=0.0
	for i in range(binsx):
		if Px[i]>0:
			E+=-Px[i]*np.log(Px[i])
	return E

def MI(x,y):

	xu=uniquelist(x)
	yu=uniquelist(y)
	binsx=int(max(x)-min(x)+1)
	binsy=int(max(y)-min(y)+1)

	Px = pdf(xu,binsx)
	Py = pdf(yu,binsy)
	Pxy = pdf([xu,yu],(binsx,binsy))
	
	
	MI=0.0
	for i in range(binsx):
		for j in range(binsy):
			if Pxy[i,j]>0:
				MI+=Pxy[i,j]*np.log(Pxy[i,j]/(Px[i]*Py[j]))
	return MI

def TE(x,y,r):

	xu=uniquelist(x)
	yu=uniquelist(y)
	L=min([len(x),len(y)])-r
	x1=xu[0:L];
	y1=yu[0:L];
	y2=yu[r:L+r];
	
	binsx=int(max(x)-min(x)+1)
	binsy=int(max(y)-min(y)+1)

	Px1y1y2 = pdf([x1,y1,y2],(binsx,binsy,binsy))
	Py1 = pdf(y1,binsy)
	Py1y2 = pdf([y1,y2],(binsy,binsy))
	Px1y1 = pdf([x1,y1],(binsx,binsy))
	
	
	TE=0.0
	for i in range(binsx):
		for j in range(binsy):
			for k in range(binsy):
				if Px1y1y2[i,j,k]>0:
					TE+=Px1y1y2[i,j,k]*np.log(Px1y1y2[i,j,k]*Py1[j]/(Py1y2[j,k]*Px1y1[i,j]))
	return TE
	
	

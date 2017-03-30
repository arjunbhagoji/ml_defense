import numpy as np
import scipy
from sklearn.metrics import pairwise
from sklearn import preprocessing

class DCA:
	def __init__(self, rho=None, rho_p=None, n_components=None):
		self.n_components = n_components
		self.rho = rho
		self.rho_p = rho_p

	def fit(self, X, y):
		(self._Sw, self._Sb) = self._get_Smatrices(X,y)

		if self.rho == None:
			s0 = np.linalg.eigvalsh(self._Sw)
			self.rho = 0.02*np.max(s0)
		if self.rho_p == None:
			self.rho_p = 0.1*self.rho

		pSw = self._Sw + self.rho*np.eye(self._Sw.shape[0])
		pSbar = self._Sb + self._Sw + (self.rho_p+self.rho)*np.eye(self._Sw.shape[0])
 		(s1,vr) = scipy.linalg.eigh(pSbar,pSw,overwrite_a=True,overwrite_b=True)
 		s1 = s1[::-1] #re-order from large to small
 		Wdca = vr.T[::-1]
		self.eigVal = s1
		self.allComponents = Wdca
		if self.n_components:
			self.components = Wdca[0:self.n_components]
		else:
			self.components = Wdca


	def transform(self, X, dim=None):
		if dim == None:
			X_trans = np.inner(self.components,X)
		else:
			X_trans = np.inner(self.allComponents[0:dim],X)
		return X_trans.T

	def inverse_transform(self, Xreduced, projMatrix=None, dim=None):
		if projMatrix is None:
			if dim is None:
				W = self.components
			else:
				W = self.allComponents[0:dim]
		else:
			W = projMatrix
		#W = PxM where P<M
		foo = np.inner(W,W)
		bar = np.linalg.solve(foo.T,W)
		Xhat = np.inner(Xreduced,bar.T)
		return Xhat

	def _get_Smatrices(self, X,y):
		Sb = np.zeros((X.shape[1],X.shape[1]))

		S = np.inner(X.T,X.T)
		N = len(X)
		mu = np.mean(X,axis=0)
		classLabels = np.unique(y)
		for label in classLabels:
			classIdx = np.argwhere(y==label).T[0]
			Nl = len(classIdx)
			xL = X[classIdx]
			muL = np.mean(xL,axis=0)
			muLbar = muL - mu
			Sb = Sb + Nl*np.outer(muLbar,muLbar)

		Sbar = S - N*np.outer(mu,mu)
		Sw = Sbar - Sb
		self.mean_ = mu

		return (Sw,Sb)

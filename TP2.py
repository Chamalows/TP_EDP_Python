#TP2
from numpy import *
from numpy.linalg import *
from numpy.random import *


Ns = 3

#Subdivision uniforme X de N+2 pts de [0,1]
X = linspace(0, 1., Ns+2)

#Calcul de la matrice A

A = 4.*eye(Ns*Ns) + -1*(diag(ones(Ns*Ns - 1), 1) + diag(ones(Ns*Ns - 1), -1))
A = A - (diag(ones(Ns*Ns - 1), 1) + diag(ones(Ns*Ns - 1), -1))


B1 = -1*(diag(ones(Ns - 1), 1) + diag(ones(Ns - 1), -1)) + 4.*eye(Ns)

B2 = -eye(Ns)

B3 = zeros((Ns, Ns))

A[[0,3],:]]

#Autre methode de creation de la matrice A

A = -1*(diag(ones(Ns*Ns-3),3) + diag(ones(Ns*Ns-3),-3))

B = 4*eye(Ns) -1*(diag(ones(Ns-1),1) + diag(ones(Ns-1),-1))

for i in arange(0,Ns):
  A[3*i:3*i+3,3*i:3*i+3] = B

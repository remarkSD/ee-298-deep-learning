import numpy as np
import matplotlib.pyplot as plt
import sys

coeffs = sys.argv
coeffs.pop(0)
coeffs = [float(x) for x in coeffs]

#print(coeffs)
power = len(coeffs)-1

'''initialization'''
A = np.linspace(-10,10, 10000)
Anew = np.ones((A.shape[0],1))
#print(Anew)
#print(A.shape)

A = A.reshape((-1,1))
for i in range (0,power):
    Anew = np.append(A**(i+1), Anew, axis=1)

'''generating output'''
B = np.matmul(Anew,coeffs)
#print(B)

'''adding noise'''
#Anoise = A + np.random.uniform(-1,1,A.shape)
Anoise = A
Bnoise = B + np.random.uniform(-1,1,B.shape)

#plotting
plt.plot(A,B,'o-')
plt.plot(Anoise,Bnoise,'o')
plt.show()


Amatrix = np.ones((Anoise.shape[0],1))
for i in range (0,3):
    #Amatrix = np.append(A**(i+1), Amatrix, axis=1)
    Amatrix = np.append(Anoise**(i+1), Amatrix, axis=1)
#print(Amatrix)
print(Amatrix)

#plotting
#plt.plot(A,B,'o-')
#plt.plot(A,Bnoise,'o')
#plt.show()
#LR = 0.000000000001
LR = 0.001
#delta = 0.01*(sum(abs(Anoise)))
delta = 1
#x = np.ones(4)
x = [0,0,0,0]

gradnorm = delta+1
loss = np.matmul(Amatrix,x) - B
loss = np.linalg.norm(loss)

ctr = 0
while gradnorm > delta:
    prevloss = loss
    loss = np.matmul(Amatrix,x) - B
    loss = np.linalg.norm(loss)

    grad = np.transpose(Amatrix)
    grad = np.matmul(grad,Amatrix)
    grad = np.matmul(grad,x)
    grad = grad - np.matmul(np.transpose(Amatrix),Bnoise)
    #print(loss)
    #print(grad)


    prevgradnorm = gradnorm
    gradnorm = np.linalg.norm(grad);

    if prevgradnorm > gradnorm:
        #if gradnorm > 5000:
        LR = LR*1.2
    elif prevgradnorm <= gradnorm:
        LR = LR*0.1

    x = x - LR*grad
    #LR = loss*0.00000000001
    print(ctr, x, gradnorm,loss, LR)
    ctr = ctr + 1

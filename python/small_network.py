import numpy as np

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_to_deriv(output):
    return output * (1 - output)

def predict(inp, weigths):
    print(inp, sigmoid(np.dot(inp, weigths)))


X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

#syn0 = np.array([[0.91850754,-0.207117,0.55211696,0.32125427],
#                 [ 0.10220112,-0.59326735,-0.16234442,0.93764421],
#                 [ 0.28220746,0.22866064,0.1132737,-0.5450892 ]])
#syn1 = np.array([[-0.67699272],[ 0.19260066],[-0.44966636],[-0.13091481]])

for j in range(6000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)


l1 = 1/(1+np.exp(-(np.dot(np.array([0,1,1]),syn0))))
l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))

print(l2)


#t = np.array([1,0,1])
#lp = 1/(1+np.exp(-(np.dot(t,syn0))))

#print(lp)

# predict([1,0,1],syn0)


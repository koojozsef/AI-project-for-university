# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

#read base images
def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict
    
dictionary = unpickle('../info/cifar-10-batches-py/data_batch_1') #loading first batch
data = dictionary['data']
labels = dictionary['labels']

dictionary = unpickle('../info/cifar-10-batches-py/data_batch_2') #loading first batch
data_test = dictionary['data']
labels_test = dictionary['labels']

"""
    SVM - Support Vector Machine
    
X[i] - input image flattened to a vector [D x 1]
    it is a 'D' dimensional point.
    
W - evaluator matrix [K x D]
    K - number of classes
    D - number of dimension of input data (X[i])
    
B - bias vector [K x 1]
"""

# initialise variables
X = data[0]
K = 10 #There are 10 classes indicated by labels
D = np.size(X)
W = np.random.randint(0,10,(K,D))
B = np.ones(K)



"""
----------------- SVM function definition -------------
"""
def SVM_Scores(x,W,b):
    #calculate scores
    scores = W.dot(x) + b
    return scores
    
#use SVM function
s1=SVM_Scores(X,W,B)
#print(s1)

"""
----------------- SVM function definition -------------
-----------------  with integrated bias   -------------
"""
#init variables
X = np.append(data[0],[1])
B = np.reshape(B,(1,K))
W = np.append(W,B.T,axis=1)

def SVM_Scores_bias(x,W):
    #calculate scores
    scores = W.dot(x)
    return scores
s2=SVM_Scores_bias(X,W)
#print(s2)

"""

----------------- Loss function

"""
def L_i(x, y, W):
  """
  unvectorized version. Compute the multiclass svm loss for a single example (x,y)
  - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
  - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
  - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
  """
  delta = 1.0 # see notes about delta later in this section
  correct_class_score = s2[y]
  loss_i = 0.0
  for j in xrange(K): # iterate over all wrong classes
    if j == y:
      # skip for the true class to only loop over incorrect classes
      continue
    # accumulate loss for the i-th example
    loss_i += max(0, s2[j] - correct_class_score + delta)
  return loss_i

loss = L_i(X,labels[0],W)
#print (loss) 
    

"""

----------------- Loss function with vectorization

"""   
def L_i_vectorized(x, y, W):
  """
  A faster half-vectorized implementation. half-vectorized
  refers to the fact that for a single example the implementation contains
  no for loops, but there is still one loop over the examples (outside this function)
  """
  delta = 1.0
  # compute the margins for all classes in one vector operation
  margins = np.maximum(0, s2 - s2[y] + delta)
  # on y-th position scores[y] - scores[y] canceled and gave delta. We want
  # to ignore the y-th position and only consider margin on max wrong class
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i
    
loss = L_i_vectorized(X,labels[0],W)
#print (loss)    

"""

----------------- Loss function with vectorization
                    For many data

"""     

def L(X,y,W,num):
    """
    fully-vectorized implementation :
  - X holds all the training examples as columns 
  - y is array of integers specifying correct class
  - W are weights (e.g. 10 x 3073)
  - num is the number of data
    """
    LOSS = np.array([])
    for i in range(num):
        l=L_i_vectorized(X[i],y[i],W)
        LOSS = np.append(LOSS,l)
        
    return LOSS #num long array

Num_of_data=100

result=L(data,labels,W,Num_of_data)
plt.plot(result[0:100])
avg_result = result.mean()

"""
--------------- Optimizations
"""
#random opt -- best ideoa
def opt_rand():
    bestloss = float("inf") # Python assigns the highest possible float value
    
    for num in xrange(100):
        rnd.seed(num)
        W_l = rnd.randn(10, 3073) * 0.0001 # generate random parameters
        loss = L(data, labels, W_l, Num_of_data).mean() # get the loss over the entire training set
        if loss < bestloss: # keep track of the best solution
            bestloss = loss
            print(loss)
            bestW = W_l
            print 'in attempt %d the loss was %f, best %f' % (num, loss, bestloss)

# gradient evaluation
def eval_numerical_gradient(f, x):
  """ 
  a naive implementation of numerical gradient of f at x 
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """ 

  fx = f(x) # evaluate function value at original point
  grad = np.zeros(x.shape)
  h = 0.00001

  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value + h # increment by h
    fxh = f(x) # evalute f(x + h)
    x[ix] = old_value # restore to previous value (very important!)

    # compute the partial derivative
    grad[ix] = (fxh - fx) / h # the slope
    it.iternext() # step to next dimension

  return grad
    
    
    
    
    
    
    
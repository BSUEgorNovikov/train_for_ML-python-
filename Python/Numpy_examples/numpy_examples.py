import numpy as np

a = np.array([[1,2], [4,5]])
b = np.array([[11,12], [7,8]])

def multiply_element_by_element():
    print(a * b)

def invariant_matrix():
    c = np.linalg.inv(a)
    print(c)

def scalar_mat_multi():
    c = np.linalg.matmul(a,b)
    print(c)

def stacking():
    c = np.vstack((a,b))
    print(c)
    d = np.hstack((a,b))
    print("\n", d)

def indexing():
    c = np.vstack((a, b))
    print(c,"\n", c[[2,1],[0,1]], "\n", c[c >= 5])

def slicing():
    c = np.vstack((a, b))
    print(c, "\n" ,c[...,1], "\n", c[2,...])

def transpose():
    c = np.vstack((a, b))
    print(c.T)

transpose()
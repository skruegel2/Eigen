import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image
from numpy import linalg as LA

def generate_w(p, n):
    ret_w = np.zeros((p, n))
    for cur_col in range(0, p):
        for cur_row in range(0,n):
            ret_w[cur_col][cur_row] = np.random.normal()
    print(ret_w.shape[0], ret_w.shape[1])
    return ret_w

def scatter_plot(W, X_tilde, X):
    plt.clf()
    plt.plot(W[0,:], W[1,:],'.')
    plt.axis('equal')
    plt.title('W Scatter Plot')
    plt.show()
    plt.clf()
    plt.plot(X_tilde[0,:], X_tilde[1,:], '.')
    plt.axis('equal')
    plt.title('X tilde Scatter Plot')
    plt.show()
    plt.clf()
    plt.plot(X[0,:], X[1,:], '.')
    plt.axis('equal')
    plt.title('X Scatter Plot')
    plt.show()
    plt.clf()

def calculate_x_tilde(W, R):
    w, v = LA.eig(R)
    print(w)
    print(v)
    l = np.array([[np.sqrt(w[0]), 0], [0, np.sqrt(w[1])]])
    print(l)
    ret_X_tilde = np.zeros_like(W)
    #for cur_col in range(0, W.shape[0]):
    #    for cur_row in range(0,W.shape[1]):
    #        ret_X_tilde[cur_col][cur_row] = W[cur_col][cur_row]*np.sqrt(w[cur_col])

    ret_X_tilde = np.dot(l, W)

    print(W[1][999],ret_X_tilde[1][999])
    #for cur_col in range(0, W.shape[0]):
    #    ret_X_tilde[cur_col] = np.sqrt(w[cur_col])*W[cur_col]
    return ret_X_tilde

def calculate_x(X_t, R):
    w, v = LA.eig(R)
    ret_X = np.zeros_like(X_t)
    ret_X = np.matmul(v, X_t)
    return ret_X

R = np.array([[2, -1.2], [-1.2, 1]])
# Generate W
W = generate_w(2, 1000)
# Calculate X_tilde
X_tilde = calculate_x_tilde(W, R)
# Calculate X
X = calculate_x(X_tilde, R)
# Produce scatter plots
scatter_plot(W, X_tilde, X)
# Estimate covariance

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image
from numpy import linalg as LA

def generate_w(n, p):
    ret_w = np.zeros((n, p))
    for cur_col in range(0, n):
        for cur_row in range(0,p):
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
    l = np.array([[np.sqrt(w[0]), 0], [0, np.sqrt(w[1])]])
    ret_X_tilde = np.zeros_like(W)
    ret_X_tilde = np.dot(l, W)
    return ret_X_tilde

def calculate_x(X_t, R):
    w, v = LA.eig(R)
    ret_X = np.zeros_like(X_t)
    ret_X = np.matmul(v, X_t)
    return ret_X

def calculate_mean(X, u_hat):
    # Sum each row
    for cur_row in range(0,X.shape[1]):
        u_hat.append(0)
        for cur_col in range(0, X.shape[0]):
            u_hat[cur_row] = u_hat[cur_row] + X[cur_col][cur_row]
        # Divide each row by the number of columns
        u_hat[cur_row] = u_hat[cur_row]/X.shape[0]
    return u_hat

def estimate_covariance(X):
    u_hat = []
    # Calculate mean
    print(X[0][0], X[1][0])
    u_hat = calculate_mean(X, u_hat)
    print(u_hat[0])
    # Calculate covariance
    #R_hat = np.zeros(X.shape[0], X.shape[0])
    #u_hat = calculate_mean(X, u_hat)
    #for cur_col in range(0, X.shape[0]):
    #    X_i = np.zeros(1, X.shape[1])
    #    for cur_row in range(0,X.shape[1]):
    #        X_i[cur_row] = X[cur_col][cur_row] - u_hat[cur_col]
    #    X_i_t  

    return X

R = np.array([[2, -1.2], [-1.2, 1]])
# Generate W
W = generate_w(2, 1000)
# Calculate X_tilde
X_tilde = calculate_x_tilde(W, R)
# Calculate X
X = calculate_x(X_tilde, R)
# Produce scatter plots
#scatter_plot(W, X_tilde, X)
# Estimate covariance
X = W
R_hat = estimate_covariance(X)
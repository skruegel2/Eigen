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
    # Sum each column
    for cur_col in range(0, X.shape[0]):
        u_hat.append(0)
        for cur_row in range(0,X.shape[1]):
            u_hat[cur_col] = u_hat[cur_col] + X[cur_col][cur_row]
        # Divide each col by the number of rows
        u_hat[cur_col] = u_hat[cur_col]/X.shape[1]
    return u_hat

def estimate_covariance(X):
    u_hat = []
    # Calculate mean
    u_hat = calculate_mean(X, u_hat)
    # Calculate Z
    Z = X
    for cur_col in range(0, X.shape[0]):
        for cur_row in range(0,X.shape[1]):
            Z[cur_col][cur_row] = Z[cur_col][cur_row] - u_hat[cur_col]
    R_hat = np.zeros((X.shape[0], X.shape[0]))
    R_hat = np.matmul(Z,np.transpose(Z))
    R_hat = R_hat * 1/(X.shape[1] - 1)  
    print(R_hat)
    return R_hat

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
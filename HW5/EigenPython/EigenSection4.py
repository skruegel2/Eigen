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
def scatter_plot(X, title):
    plt.clf()
    plt.plot(X[0,:], X[1,:],'.')
    plt.axis('equal')
    plt.title(title)
    plt.show()

def scatter_plot_three(W, X_tilde, X):
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
    return R_hat

def compute_X_hat(X, R_hat):
    w, v = LA.eig(R_hat)
    X_hat = np.matmul(np.transpose(v), X)
    return X_hat

def compute_W(X_hat, R_hat):
    w, v = LA.eig(R_hat)
    w = np.power(w, -0.5)
    l = np.array([[w[0], 0], [0, w[1]]])
    W_new = np.dot(l, X_hat)
    return W_new

def subtract_mean(X):
    u_hat = []
    # Calculate mean
    u_hat = calculate_mean(X, u_hat)
    for cur_col in range(0, X.shape[0]):
        for cur_row in range(0,X.shape[1]):
            X[cur_col][cur_row] = X[cur_col][cur_row] - u_hat[cur_col]
    return X

# The following are strings used to assemble the data file names
datadir='./training_data'    # directory where the data files reside
dataset=['arial','bookman_old_style','century','comic_sans_ms','courier_new',
  'fixed_sys','georgia','microsoft_sans_serif','palatino_linotype',
  'shruti','tahoma','times_new_roman']
datachar='abcdefghijklmnopqrstuvwxyz'

def read_data():
    """
        Read in all these training images into columns of a single matrix X.
    
        Returns:
            X: Image column matrix.
    
    """
    Rows=64    # all images are 64x64
    Cols=64
    n=len(dataset)*len(datachar)  # total number of images
    p=Rows*Cols   # number of pixels

    X=np.zeros((p,n))  # images arranged in columns of X
    k=0
    for dset in dataset:
        for ch in datachar:
            fname='/'.join([datadir,dset,ch])+'.tif'
            im=Image.open(fname)
            img = np.array(im)
            X[:,k]=np.reshape(img,(1,p))
            k+=1
    return X

# display samples of the training data
def display_samples(X,ch):
    """
    Display samples.

    Args:
    X (ndarray) : Image column matrix.
    ch (char) : A char 'a'~'z'.

    Returns:

    """
    ind = ord(ch)-ord('a')
    fig, axs = plt.subplots(3, 4)
    for k in range(len(dataset)):
        img=np.reshape(X[:,26*(k-1)+ind],(64,64))

        axs[k//4,k%4].imshow(img,cmap=plt.cm.gray, interpolation='none') 
        axs[k//4,k%4].set_title(dataset[k])

def display_sample_at_index(X,index):
    """
    Display samples.

    Args:
    X (ndarray) : Image column matrix.
    index: index of image

    Returns:

    """
    fig, axs = plt.subplots(3, 4)
    for k in range(12):
        img=np.reshape(X[:,index+k],(64,64))

        axs[k//4,k%4].imshow(img,cmap=plt.cm.gray, interpolation='none') 
        axs[k//4,k%4].set_title(k)

def divide_n_1(X):
    X = X * 1/(np.sqrt(X.shape[1] - 1))
    return X

def plot_12_largest_values(Z):
    U, s, vh = np.linalg.svd(Z,full_matrices = True)
    print(U.shape[0],U.shape[1])
    display_sample_at_index(U,0)
    plt.show()
    #Y = np.matmul(np.transpose(U),Z)
    #X_hat = np.matmul(U,Y)
    #display_sample_at_index(X_hat,0)
    #plt.show()

X = read_data()
#display_samples(X,'a')
#plt.show()
X = subtract_mean(X)
Z = divide_n_1(X)
plot_12_largest_values(Z)
#R = np.array([[2, -1.2], [-1.2, 1]])
## Generate W
#W = generate_w(2, 1000)
## Calculate X_tilde
#X_tilde = calculate_x_tilde(W, R)
## Calculate X
#X = calculate_x(X_tilde, R)
## Produce scatter plots
##scatter_plot_three(W, X_tilde, X)
## Estimate covariance
#R_hat = estimate_covariance(X)
##print(R_hat)
## Compute X hat
#X_hat = compute_X_hat(X, R_hat)
## Compute W new
#W_new = compute_W(X_hat, R_hat)
#scatter_plot(X_hat, 'X Hat')
#scatter_plot(W_new, 'W')
## Estimate covariance
#R_w = estimate_covariance(W_new)
#print(R_w)

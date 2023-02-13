import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image

def equalize(X):
    x = np.array(X)
    print(x.shape[0], x.shape[1])
    F_x = np.ndarray(shape=(256,))
    F_xtot = np.ndarray(shape=(256,))
    N,test_bins,patches = plt.hist(x.flatten(), bins=np.linspace(0,255,256))
    F_total = 0
    for cur_bin in range(0,255):
        F_total = F_total + N[cur_bin]
        F_xtot[cur_bin] = F_total
    # Handle last element.  For some reason N does not extend to element 255
    #print('F_total', F_total)
    for cur_bin in range(0,256):
        F_x[cur_bin] = F_xtot[cur_bin]/F_total
    x_axis = np.linspace(0,255,256)
    F_x[255] = 1.0
##    plt.clf()
##    plt.xlabel("i")
##    plt.ylabel("Fx[i]")
##    plt.title("Cumulative Distribution Function")    
##    plt.plot(F_x)
##    plt.show()
    # Find Ymin and Ymax.  Set them to the max, min respectively
    Ymin = 1.0
    Ymax = 0.0
    image_len = x.flatten().size
    x_flat = x.flatten()
    #print(image_len)
    for cur_pixel in range(0,image_len):
        if (Ymin > F_x[x_flat[cur_pixel]]):
            Ymin = F_x[x_flat[cur_pixel]]
        if (Ymax < F_x[x_flat[cur_pixel]]):
            Ymax = F_x[x_flat[cur_pixel]]
    print('Ymin:', Ymin)
    print('Ymax:', Ymax)
    #Initialize the equalized image
    x_eq = np.array(X)
    num_rows = x.shape[0]
    num_cols = x.shape[1]
    for cur_row in range(0,num_rows):
         for cur_col in range(0, num_cols):
             x_eq[cur_row][cur_col] = (255*(F_x[x[cur_row][cur_col]] - Ymin))/(Ymax -Ymin)
    gray = cm.get_cmap('gray', 256)
    plt.clf()
    plt.imshow(x_eq, cmap=gray);
    plt.show()
        
gray = cm.get_cmap('gray', 256)
im = Image.open('kids.tif')
equalize(im)
##plt.clf()
##eq_im = Image.open('Equalized kids.tif')
##equalized_x = np.array(eq_im)
##plt.hist(equalized_x.flatten(), bins=np.linspace(0,255,256))
##plt.xlabel("Pixel Intensity")
##plt.ylabel("Number of Pixels")
##plt.title("Equalized Kids.tif Histogram")
###plt.imshow(x, cmap=gray);
##plt.show()

##gray = cm.get_cmap('gray', 256)
##im = Image.open('kids.tif')
##x = np.array(im)
             
##plt.hist(x.flatten(), bins=np.linspace(0,255,256))
##plt.xlabel("Pixel Intensity")
##plt.ylabel("Number of Pixels")
##plt.title("Kids.tif Histogram")
#plt.imshow(x, cmap=gray);

#plt.show()
##fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
##
### Make data.
##u = np.arange(-np.pi, np.pi, 0.1)
##v = np.arange(-np.pi, np.pi, 0.1)
##u,v = np.meshgrid(u, v)
###Z = np.absolute(0.01/(((1-0.9*(np.cos(u)-1j*np.sin(u)))*(1-0.9*(np.cos(v)-1j*np.sin(v))))))
###Z = np.exp(u*1j)
##Z = np.log((1.0/12.)*((3/((1-0.99*np.exp(u*-1j)-0.99*np.exp(v*-1j)+0.9801*np.exp((u+v)*-1j))))**2))
### Plot the surface.
##surf = ax.plot_surface(u, v, Z, cmap=cm.coolwarm,
##                       linewidth=0, antialiased=False)
##
### Customize the z axis.
###ax.set_zlim(-1.01, 1.01)
##ax.zaxis.set_major_locator(LinearLocator(10))
### A StrMethodFormatter is used automatically
##ax.zaxis.set_major_formatter('{x:.02f}')
##
### Add a color bar which maps values to colors.
##fig.colorbar(surf, shrink=0.5, aspect=5)
##
##plt.show()

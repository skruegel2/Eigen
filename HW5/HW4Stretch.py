import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image

def stretch(input_img, T1, T2):
    # Step through rows and columns translating the gray levels
    x = np.array(input_img)
    num_rows = x.shape[0]
    num_cols = x.shape[1]
    output_img = np.array(input_img)
    for cur_row in range(0,num_rows):
         for cur_col in range(0, num_cols):
             if (x[cur_row][cur_col] < T1):
                 output_img[cur_row][cur_col] = 0
             elif (x[cur_row][cur_col] > T2):   
                 output_img[cur_row][cur_col] = 255
             else:
                 output_img[cur_row][cur_col] = round((x[cur_row][cur_col] - T1) * (253/(T2 - T1)))
             #print('cur_row:', cur_row, 'cur_col:', cur_col, output_img[cur_row][cur_col])
             # Boundary check
             if (output_img[cur_row][cur_col] < 0):
                 output_img[cur_row][cur_col] = 0
             elif (output_img[cur_row][cur_col] > 255):
                 output_img[cur_row][cur_col] = 255
    return output_img

# Open and display kids.tif        
gray = cm.get_cmap('gray', 256)
im = Image.open('kids.tif')
im.show();
#Initialize the equalized image
x_eq = np.array(im)
# Histogram
plt.clf()
plt.hist(x_eq.flatten(), bins=np.linspace(0, 255, 256))
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.title("Kids.tif Histogram")
plt.show()
plt.clf()
output = stretch(im, 70, 180)
plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.imshow(output, cmap=gray)
plt.show()
# Stretch histogram
plt.clf()
plt.hist(output.flatten(), bins=np.linspace(0, 255, 256))
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.title("Stretch Kids.tif Histogram")
plt.show()
plt.clf()


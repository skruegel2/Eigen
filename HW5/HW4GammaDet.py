import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image

g = 155
def create_gray_stripe(pattern, stripe, graylevel):
    for row in range(0,16):
        for col in range(0,256):
            pattern[stripe*16+row][col] = graylevel

    return pattern

def create_check_stripe(pattern, stripe):
    check_block = np.zeros((4,4),np.uint8)
    check_block[0][0] = 255;
    check_block[0][1] = 255;
    check_block[0][2] = 0;
    check_block[0][3] = 0;
    check_block[1][0] = 255;
    check_block[1][1] = 255;
    check_block[1][2] = 0;
    check_block[1][3] = 0;
    check_block[2][0] = 0;
    check_block[2][1] = 0;
    check_block[2][2] = 255;
    check_block[2][3] = 255;
    check_block[3][0] = 0;
    check_block[3][1] = 0;
    check_block[3][2] = 255;
    check_block[3][3] = 255;
    for stripe_row in range(0, 4):
        for stripe_col in range(0,256):
            if (stripe_col % 4 == 0):
                for row in range(0,4):
                    for col in range(0,4):
                        pattern[stripe*16+4*stripe_row+row][stripe_col+col] = check_block[row][col]
    return pattern


pattern = np.zeros((256, 256),np.uint8)
for stripe in range(0,16):
    if (stripe % 2 == 0):
        pattern = create_check_stripe(pattern, stripe)
    else:
        pattern = create_gray_stripe(pattern, stripe, g)
plt.title("")
plt.xlabel("")
plt.ylabel("")
gray = cm.get_cmap('gray', 256)
plt.imshow(pattern, cmap=gray)
plt.show()
# Open and display kids.tif        
#gray = cm.get_cmap('gray', 256)
#im = Image.open('kids.tif')
#im.show();
##Initialize the equalized image
#x_eq = np.array(im)
## Histogram
#plt.clf()
#plt.hist(x_eq.flatten(), bins=np.linspace(0, 255, 256))
#plt.xlabel("Pixel Intensity")
#plt.ylabel("Number of Pixels")
#plt.title("Kids.tif Histogram")
#plt.show()
#plt.clf()
#output = stretch(im, 70, 180)
#plt.title("")
#plt.xlabel("")
#plt.ylabel("")
#plt.imshow(output, cmap=gray)
#plt.show()
## Stretch histogram
#plt.clf()
#plt.hist(output.flatten(), bins=np.linspace(0, 255, 256))
#plt.xlabel("Pixel Intensity")
#plt.ylabel("Number of Pixels")
#plt.title("Stretch Kids.tif Histogram")
#plt.show()
#plt.clf()


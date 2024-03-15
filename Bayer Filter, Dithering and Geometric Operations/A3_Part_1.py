# import statements
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage import io

def part1():
    """ BasicBayer: reconstruct RGB image using GRGB pattern"""
    filename_Grayimage = 'PeppersBayerGray.bmp'

    # read image
    img = io.imread(filename_Grayimage, as_gray =True)
    h,w = img.shape

    # our final image will be a 3 dimentional image with 3 channels
    rgb = np.zeros((h,w,3),np.uint8)

    # reconstruction of the green channel IG
    IG = np.copy(img) # copy the image into each channel

    for row in range(0,h,4): # loop step is 4 since our mask size is 4.
        for col in range(0,w,4): # loop step is 4 since our mask size is 4.
            # TODO: compute pixel value for each location where mask is unshaded (0) 
            # interpolate each pixel using its every valid (shaded) neighbour
            IG[row,col+1]= (int(img[row,col])+int(img[row,col+2])+int(img[row+1,col+1]))/3  # B (A + C + F)/3
            IG[row, col+3]= (int(img[row,col+2])+int(img[row+1,col+3]))/2    # D (C + H)/2
            IG[row+1,col]= (int(img[row,col])+int(img[row+1,col+1])+int(img[row+2,col]))/3  # E (A + F + I)/3
            IG[row+1, col+2]= (int(img[row+1,col+1])+int(img[row,col+2])+int(img[row+1,col+3]+int(img[row+2,col+2])))/4  # G (F + C + H + K)/4
            IG[row+2,col+1]= (int(img[row+1,col+1])+int(img[row+2,col])+int(img[row+2,col+2])+int(img[row+3,col+1]))/4  # J (F + I + K + N)/4
            IG[row+2,col+3]= (int(img[row+1,col+3])+int(img[row+2,col+2])+int(img[row+3,col+3]))/3  # L (H + K+ P)/3
            IG[row+3,col]= (int(img[row+2,col])+int(img[row+3,col+1]))/2     # M (J + N)/2
            IG[row+3,col+2]= (int(img[row+2,col+2])+int(img[row+3,col+1])+int(img[row+3,col+3]))/3  # O (K + N + P)/3
            

    # TODO: show green (IR) in first subplot (221) and add title - refer to rgb one for hint on plotting
    plt.figure(figsize=(10,8))
    plt.subplot(221)
    plt.imshow(IG, cmap='gray'), plt.title('IG')


    # reconstruction of the red channel IR
    IR = np.copy(img)

    for row in range(0,h,4):
        for col in range(0,w,4):
            IR[row,col]= (int(img[row,col+1]))  # A B
            IR[row,col+2]= (int(img[row,col+1])+int(img[row,col+3]))/2  # C (B + D)/2
            IR[row+1,col]= (int(img[row,col+1])+int(img[row+2,col+1]))/2  # E (B + J)/2
            IR[row+1,col+1]= (int(img[row,col+1])+int(img[row+2,col+1]))/2  # F (B + J)/2
            IR[row+1,col+2]=(int(img[row,col+1])+int(img[row,col+3])+int(img[row+2,col+3])+int(img[row+2,col+1]))/4  # G (B + D + J + L)/4
            IR[row+2,col]= (int(img[row+2,col+1]))  # I J
            IR[row+3,col]= (int(img[row+2,col+1])) # M J
            IR[row+3,col+1]= (int(img[row+2,col+1])) # N J
            IR[row+1,col+3]= (int(img[row,col+3])+int(img[row+2,col+3]))/2  # H (D + L)/2
            IR[row+2,col+2]= (int(img[row+2,col+1])+int(img[row+2,col+3]))/2  # K (J + L)/2
            IR[row+3,col+2]= (int(img[row+2,col+1])+int(img[row+2,col+3]))/2  # O (J + L)/2
            IR[row+3,col+3]= (int(img[row+2,col+3]))  # P L

    # show red (IR) in the second subplot (222) and add title
    plt.subplot(222)
    plt.imshow(IR, cmap='gray'), plt.title('IR')

    # reconstruction of the blue channel IB
    IB = np.copy(img)

    for row in range(0,h,4):
        for col in range(0,w,4):
            IB[row,col]= int(img[row+1,col])  # A E
            IB[row,col+1]= (int(img[row+1,col])+int(img[row+1,col+2]))/2 # B (E + G)/2
            IB[row,col+2]= (int(img[row+1,col+2]))  # C G
            IB[row,col+3]= (int(img[row+1,col+2]))  # D G
            IB[row+1,col+3]= (int(img[row+1,col+2]))  # H G
            IB[row+1,col+1]= (int(img[row+1,col])+int(img[row+1,col+2]))/2  # F (E + G)/2
            IB[row+2,col]= (int(img[row+1,col])+int(img[row+3,col]))/2 # I (E + M)/2
            IB[row+2,col+1]= (int(img[row+1,col])+int(img[row+1,col+2])+int(img[row+3,col+2])+int(img[row+3,col]))/4 # J (E + G + O + M)/4
            IB[row+2,col+2]= (int(img[row+1,col+2])+int(img[row+3,col+2]))/2  # K (G + O)/2
            IB[row+2,col+3]= (int(img[row+1,col+2])+int(img[row+3,col+2]))/2   # L (G + O)/2
            IB[row+3,col+1]= (int(img[row+3,col])+int(img[row+3,col+2]))/2  # N (M + O)/2
            IB[row+3,col+3]= (int(img[row+3,col+2])) #P O

    # show blue (IB) in the third subplot (223) and add title
    plt.subplot(223)
    plt.imshow(IB, cmap='gray'), plt.title('IB')

    # merge the three channels IG, IR, IB in the correct order
    rgb[:, :, 1] = IG
    rgb[:, :, 0] = IR
    rgb[:, :, 2] = IB


    # TODO: show rgb image in final subplot (224) and add title
    plt.subplot(224)
    plt.imshow(rgb),plt.title('rgb')
    plt.show()

if __name__  == "__main__":
    part1()
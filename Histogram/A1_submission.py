"""Include your imports here
Some example imports are below"""

import numpy as np 
from skimage import io, img_as_ubyte
import matplotlib.pyplot as plt
import math

# Histogram computed by your code (cannot use in-built functions!)
def compute_histogram(image, n_bins):
    img = img_as_ubyte(image)

    hist = np.zeros(n_bins, dtype=int)
    pixels = 256 / n_bins

    for i in img.flatten():
        index = int(i // pixels)
        hist[index] += 1

    return hist


def part1_histogram_compute():
    filename = r'test.jpg'
    image = io.imread(filename, as_gray=True)
    img = img_as_ubyte(image)

    """add your code here"""
    n = 64

    hist = compute_histogram(image, n)

    # Histogram computed by numpy
    hist_np, _ = np.histogram(img.flatten(), bins=n, range=[0, 256])

    """Plotting code provided here
    Make sure to match the variable names in your code!"""

    plt.figure(figsize=(8, 10))
    plt.subplot(121), plt.plot(hist), plt.title('My Histogram')
    plt.xlim([0, n])
    plt.subplot(122), plt.plot(hist_np), plt.title('Numpy Histogram')
    plt.xlim([0, n])

    plt.show()


def histogram_equalization(img, n_bins):
    hist, bins = np.histogram(img.flatten(), bins=n_bins, range=[0, 256])

    K = 256  # Number of possible pixel values

    img_eq = np.zeros_like(img, dtype=np.uint8)

    M, N = img.shape
    total_pixels = M * N

    for i in range(n_bins):
        hist_sum = np.sum(hist[:i + 1])
        img_eq[(img >= bins[i]) & (img <= bins[i + 1])] = int((K - 1) * hist_sum / total_pixels + 0.5)

    hist_eq, _ = np.histogram(img_eq.flatten(), bins=n_bins, range=[0, 256])

    return img_eq, hist_eq

def part2_histogram_equalization():
    filename = r'test.jpg'
    image = io.imread(filename, as_gray=True)
    img = img_as_ubyte(image)

    n_bins = 64

    hist = compute_histogram(image, n_bins)
    img_eq1, hist_eq = histogram_equalization(img, n_bins)

    # Plotting code
    plt.figure(figsize=(8, 10))
    plt.subplot(221), plt.imshow(image, 'gray'), plt.title('Original Image')
    plt.subplot(222), plt.plot(hist), plt.title('Histogram')
    plt.xlim([0, n_bins])
    plt.subplot(223), plt.imshow(img_eq1, 'gray'), plt.title('New Image')
    plt.subplot(224), plt.plot(hist_eq), plt.title('Histogram After Equalization')
    plt.xlim([0, n_bins])

    plt.show()


def part3_histogram_comparing():

    filename1 = 'day.jpg'
    filename2 = 'night.jpg'

    # Read in the image
    img1 = io.imread(filename1, as_gray=True)
    # Read in another image
    img2 = io.imread(filename2, as_gray=True)
    
    # Calculate the histograms for img1 and img2 (you can use skimage or numpy)
    hist1, _ = np.histogram(img1.flatten(), bins=256, range=[0, 1])
    hist2, _ = np.histogram(img2.flatten(), bins=256, range=[0, 1])

    # Normalize the histograms for img1 and img2
    hist1_norm = hist1 / np.sum(hist1)
    hist2_norm = hist2 / np.sum(hist2)

    # Calculate the Bhattacharya coefficient (check the wikipedia page linked on eclass for formula)
    bc = np.sum(np.sqrt(hist1_norm * hist2_norm))

    print("Bhattacharyya Coefficient: ", bc)


def histogram_matching(image, reference):
        # Step 1: Compute PA(a) - normalized cumulative input histogram
        hist_input, _ = np.histogram(image.flatten(), bins=256, range=[0, 256], density=True)
        PA = np.cumsum(hist_input)

        # Step 2: Compute PR(a) - normalized cumulative reference histogram
        hist_ref, _ = np.histogram(reference.flatten(), bins=256, range=[0, 256], density=True)
        PR = np.cumsum(hist_ref)

        # Step 3: Compute the inverse mapping function A[a], = invPR(PA[a])
        A = np.zeros(256)
        for a in range(256):
            a_prime = 0
            while a_prime < 255 and PA[a] > PR[a_prime]:
                a_prime += 1
            A[a] = a_prime

        # Step 4: Apply the mapping to each pixel in the image
        matched_image = A[image]

        return matched_image


def part4_histogram_matching():
    filename1 = 'day.jpg'
    filename2 = 'night.jpg'

    #============Grayscale============

    # Read in the image
    source_gs = io.imread(filename1, as_gray=True)
    source_gs = img_as_ubyte(source_gs)
    # Read in another image
    template_gs = io.imread(filename2, as_gray=True)
    template_gs = img_as_ubyte(template_gs)

    # Grayscale histogram matching
    matched_gs = histogram_matching(source_gs, template_gs)

    fig = plt.figure()
    gs = plt.GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)

    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(source_gs, cmap=plt.cm.gray)
    ax1.set_title('source_gs')
    ax2.imshow(template_gs, cmap=plt.cm.gray)
    ax2.set_title('template_gs')
    ax3.imshow(matched_gs, cmap=plt.cm.gray)
    ax3.set_title('matched_gs')
    plt.show()

    #============RGB============

    # Read in the image
    source_rgb = io.imread(filename1)
    # Read in another image
    template_rgb = io.imread(filename2)

    # RGB histogram matching
    matched_rgb = np.zeros_like(source_rgb)
    for channel in range(3):
        matched_rgb[:, :, channel] = histogram_matching(source_rgb[:, :, channel], template_rgb[:, :, channel])

    fig = plt.figure()
    gs = plt.GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)

    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(source_rgb)
    ax1.set_title('source_rgb')
    ax2.imshow(template_rgb)
    ax2.set_title('template_rgb')
    ax3.imshow(matched_rgb)
    ax3.set_title('matched_rgb')
    plt.show()

if __name__ == '__main__':
    part1_histogram_compute()
    part2_histogram_equalization()
    part3_histogram_comparing()
    part4_histogram_matching()

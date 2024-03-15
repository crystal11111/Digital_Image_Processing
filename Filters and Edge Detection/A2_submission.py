import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, feature
from scipy import signal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import cosine


def filter_image(image, kernel):
    # Get kernel size
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the input image with zeros
    padded_image = np.pad(image, pad_width=((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    # Get dimensions
    height, width = image.shape

    # Initialize filtered image
    filtered_image = np.zeros((height, width))

    # Perform convolution with the given kernel
    for i in range(pad_height, height + pad_height):
        for j in range(pad_width, width + pad_width):
            window = padded_image[i-pad_height:i+pad_height+1, j-pad_width:j+pad_width+1]
            filtered_image[i-pad_height, j-pad_width] = np.sum(window * kernel)

    return filtered_image


def part1():
    # Read the grayscale image
    image = plt.imread('moon.png').astype(float)

    # 1. Laplacian Filter
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplacian_result = filter_image(image, laplacian_kernel)

    # Normalize the Laplacian response
    #min_r = np.min(laplacian_result)
    #max_r = np.max(laplacian_result)
    #laplacian_result = (laplacian_result - min_r) / (max_r - min_r)


    # 2. Gaussian Filter
    gaussian_kernel = (1/273) * np.array([[1, 4, 7, 4, 1],
                                        [4, 16, 26, 16, 4],
                                        [7, 26, 41, 26, 7],
                                        [4, 16, 26, 16, 4],
                                        [1, 4, 7, 4, 1]])
    gaussian_filtered = filter_image(image, gaussian_kernel)


    # 3. Custom Filters
    custom_filter1 = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 0, 1, 0],
                            [0, 0, 0, 1, 0]])
    custom_result1 = filter_image(image, custom_filter1)

    # 4. Custom Filters
    custom_filter2 = np.array([[0, 0, 0],
                            [6, 0, 6],
                            [0, 0, 0]])

    custom_result2 = filter_image(image, custom_filter2)

    # 5/6 Image Enhancement
    laplace_enhanced = image - laplacian_result
    laplace_enhanced = np.clip(laplace_enhanced, 0, 1)

    gaussian_enhanced = image + (image - gaussian_filtered)


    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(laplacian_result, cmap='gray'), plt.title('Laplace filtered image')
    plt.xticks([]), plt.yticks([])

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(gaussian_filtered, cmap='gray'), plt.title('Gaussian filtered image')
    plt.xticks([]), plt.yticks([])

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(custom_result1, cmap='gray'), plt.title('filtered image')
    plt.xticks([]), plt.yticks([])

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(custom_result2, cmap='gray'), plt.title('filtered image')
    plt.xticks([]), plt.yticks([])

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(laplace_enhanced, cmap='gray'), plt.title('Enhanced image')
    plt.xticks([]), plt.yticks([])

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(gaussian_enhanced, cmap='gray'), plt.title('Gaussian enhanced image')
    plt.xticks([]), plt.yticks([])

    plt.show()



def part2():
    noisy_image = io.imread('noisy.jpg')

    # Convert the image to grayscale if it's in color
    if len(noisy_image.shape) == 3:
        noisy_image = color.rgb2gray(noisy_image)

    # Apply Median filter
    median_filtered = filters.median(noisy_image)

    # Apply Gaussian filter
    gaussian_filtered = filters.gaussian(noisy_image, sigma=1)

    plt.figure(figsize=(10, 5))

    plt.subplot(131)
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Original')

    plt.subplot(132)
    plt.imshow(median_filtered, cmap='gray')
    plt.title('Median')

    plt.subplot(133)
    plt.imshow(gaussian_filtered, cmap='gray')
    plt.title('Gaussian')

    plt.show()


def inpaint_image(damaged_image, damage_mask):
    num_iterations = 0
    damaged_image = damaged_image.squeeze()
    prev_image = damaged_image.copy()
    change = np.inf
    tolerance = 0

    while change > tolerance:
        # Step (a): Blur the entire damaged image with a Gaussian smoothing filter
        blurred_image = filters.gaussian(damaged_image, sigma=2)

        # Step (b): Replace undamaged pixels with original pixels using the mask
        damaged_image = damage_mask * damaged_image + (1 - damage_mask) * blurred_image

        # Update the damaged image with the inpainted result
        change = np.mean(np.abs(damaged_image - prev_image))
        prev_image = damaged_image.copy()
        
        num_iterations += 1

    return damaged_image, num_iterations

def part3():
    damaged_image = io.imread('damage_cameraman.png')
    damage_mask = io.imread('damage_mask.png', as_gray=True)

    # Convert to grayscale if the image is in color
    if damaged_image.shape[-1] == 3:
        damaged_image = color.rgb2gray(damaged_image)

    # Normalize mask to binary values (0 or 1)
    damage_mask = (damage_mask > 0.5).astype(np.uint8)

    # Display the original damaged image
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(damaged_image, cmap='gray')
    plt.title('Damaged Image')

    # Apply inpainting
    inpainted_image, num_iterations = inpaint_image(damaged_image, damage_mask)

    # Display the inpainted image
    plt.subplot(1, 2, 2)
    plt.imshow(inpainted_image, cmap='gray')
    plt.title('Restored Image')

    plt.show()
    


def part4():
    image = io.imread('ex2.jpg', as_gray=True)

    # Display the original image
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    # Define Sobel filters for horizontal and vertical derivatives
    sobel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Compute horizontal derivative
    derivative_horizontal = signal.convolve2d(image, sobel_horizontal, mode='same', boundary='symm')

    # Display horizontal derivative image
    ax1 = plt.subplot(2, 2, 2)
    im1 = plt.imshow(derivative_horizontal, cmap='gray')
    ax1.set_title('Horizontal Gradient')

    # Compute vertical derivative
    derivative_vertical = signal.convolve2d(image, sobel_vertical, mode='same', boundary='symm')

    # Display vertical derivative image
    ax2 = plt.subplot(2, 2, 3)
    im2 = plt.imshow(derivative_vertical, cmap='gray')
    ax2.set_title('Vertical Gradien')

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(derivative_horizontal**2 + derivative_vertical**2)

    # Display gradient magnitude image
    ax3 = plt.subplot(2, 2, 4)
    im3 = plt.imshow(gradient_magnitude, cmap='gray')
    ax3.set_title('Gradient Magnitude')

    plt.show()


def part5():
    image = io.imread('ex2.jpg', as_gray=True)

    smoothed_image = filters.gaussian(image, sigma=1.5)

    target_image = io.imread('canny_target.jpg', as_gray=True)
    best_distance = 1e10
    best_params = [0, 0, 0]

    low_threshold = [50, 70, 90]
    high_threshold = [150, 170, 190]
    sigmas = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]

    for low_thresh in low_threshold:
        for high_thresh in high_threshold:
            for sigma in sigmas:
                canny_output = feature.canny(image, low_threshold=low_thresh, high_threshold=high_thresh, sigma=sigma)
                if np.sum(canny_output>0.0)>0.0:
                    this_dist = cosine(canny_output.flatten(), target_image.flatten())
                    if this_dist < best_distance:
                        best_distance = this_dist
                        best_params = [low_thresh, high_thresh, sigma]
    
    my_image = feature.canny(image, low_threshold=best_params[0], high_threshold=best_params[1], sigma=best_params[2])

    plt.figure(figsize=(10, 5))

    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(smoothed_image, cmap='gray')
    plt.title(f'Smoothed Image')

    plt.subplot(2, 2, 3)
    plt.imshow(target_image, cmap='gray')
    plt.title('Target Image')

    plt.subplot(2, 2, 4)
    plt.imshow(my_image, cmap='gray')
    plt.title('My Image')

    print(f'Cosine Distance: {best_distance:.4f}')

    plt.show()




if __name__ == '__main__':
    part1()
    part2()
    part3()
    part4()
    part5()




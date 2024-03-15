# Import libraries
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import warp, AffineTransform

def read_image():
    original_img = io.imread('bird.jpeg')
    return original_img


def calculate_trans_mat(image):
    """
    return translation matrix that shifts center of image to the origin and its inverse
    """
    rows, cols = image.shape[:2]

    tx = cols / 2
    ty = rows / 2

    trans_mat = np.array([[1, 0, -tx],
                          [0, 1, -ty],
                          [0, 0, 1]])
    
    trans_mat_inv = np.linalg.inv(trans_mat)
    
    return trans_mat, trans_mat_inv



def rotate_image(image):
    ''' rotate and return image '''
    h, w = image.shape[:2]
    trans_mat, trans_mat_inv = calculate_trans_mat(image)

    # TODO: determine angle and create Tr
    angle = 75
    angle_rad = np.radians(angle)
    Tr = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]])
    
    # Combine the translation matrix with the rotation matrix
    Tr_combined = np.dot(trans_mat_inv, np.dot(Tr, trans_mat))

    out_img = np.zeros_like(image)
    for out_y in range(h):
        for out_x in range(w):
            # TODO: find input pixel location from output pixel ocation and inverse transform matrix, copy over value from input location to output location
            input_coords = np.dot(Tr_combined, np.array([out_x, out_y, 1]))
            in_x, in_y = input_coords[:2]

            if 0 <= in_x < w and 0 <= in_y < h:
                out_img[out_y, out_x] = image[int(in_y), int(in_x)]

    return out_img, Tr



def scale_image(image):
    ''' scale image and return '''
    # TODO: implement this function, similar to above
    out_img = np.zeros_like(image)

    h, w = image.shape[:2]
    trans_mat, trans_mat_inv = calculate_trans_mat(image)
    
    scale_factor_x = 1 / 1.5
    scale_factor_y = 1 / 2.5

    Ts = np.array([[scale_factor_x, 0, 0],
                   [0, scale_factor_y, 0],
                   [0, 0, 1]])
    
    Ts_combined = np.dot(trans_mat_inv, np.dot(Ts, trans_mat))

    for out_y in range(h):
        for out_x in range(w):
            # Find input pixel location from output pixel location and combined transform matrix
            input_coords = np.dot(Ts_combined, np.array([out_x, out_y, 1]))
            in_x, in_y = input_coords[:2]

            # Check if the transformed coordinates are within the bounds of the original image
            if 0 <= in_x < w and 0 <= in_y < h:
                out_img[out_y, out_x] = image[int(in_y), int(in_x)]

    return out_img, Ts



def skew_image(image):
    ''' Skew image and return '''
    # TODO: implement this function like above
    out_img = np.zeros_like(image)
    h, w = image.shape[:2]

    trans_mat, trans_mat_inv = calculate_trans_mat(image)

    skew_param = -0.2
    Tskew = np.array([[1, skew_param, 0],
                      [skew_param, 1, 0],
                      [0, 0, 1]])
    
    Tskew_combined = np.dot(trans_mat_inv, np.dot(Tskew, trans_mat))

    for out_y in range(h):
        for out_x in range(w):
            input_coords = np.dot(Tskew_combined, np.array([out_x, out_y, 1]))
            in_x, in_y = input_coords[:2]

            if 0 <= in_x < w and 0 <= in_y < h:
                out_img[out_y, out_x] = image[int(in_y), int(in_x)]

    return out_img, Tskew


def combined_warp(image):
    # Get transformation matrices for rotation, scaling, and skew
    _, Tr = rotate_image(image)
    _, Ts = scale_image(image)
    _, Tskew = skew_image(image)

    # Combine the transformation matrices
    Tc = np.dot(Tskew, np.dot(Tr, Ts))

    # Apply the combined transformation to the image
    out_img = np.zeros_like(image)
    h, w = image.shape[:2]
    trans_mat, trans_mat_inv = calculate_trans_mat(image)

    Tcombined = np.dot(trans_mat_inv, np.dot(Tc, trans_mat))

    for out_y in range(h):
        for out_x in range(w):
            # Find input pixel location from output pixel location and combined transform matrix
            input_coords = np.dot(Tcombined, np.array([out_x, out_y, 1]))
            in_x, in_y = map(int, input_coords[:2])

            # Check if the transformed coordinates are within the bounds of the original image
            if 0 <= in_x < w and 0 <= in_y < h:
                out_img[out_y, out_x] = image[in_y, in_x]

    return out_img, Tc



def combined_warp_biinear(image):
    ''' perform the combined warp with bilinear interpolation using skimage.transform.warp '''
    # Get transformation matrices for rotation, scaling, and skew
    _, Tr = rotate_image(image)
    _, Ts = scale_image(image)
    _, Tskew = skew_image(image)

    Tc = np.dot(Tskew, np.dot(Tr, Ts))

    # Combine the transformation matrices
    out_img = np.zeros_like(image)
    trans_mat, trans_mat_inv = calculate_trans_mat(image)

    Tcombined = np.dot(trans_mat_inv, np.dot(Tc, trans_mat))

    combined_transform = AffineTransform(matrix=Tcombined)

    # Apply the combined transformation to the image using bilinear interpolation
    out_img = warp(image, inverse_map=combined_transform, order=1, mode='constant', cval=0, output_shape=image.shape, preserve_range=True)

    return out_img



if __name__ == "__main__":
    image = read_image()
    plt.imshow(image), plt.title("Oiginal Image"), plt.show()

    rotated_img, _ = rotate_image(image)
    plt.figure(figsize=(15,5))
    plt.subplot(131),plt.imshow(rotated_img), plt.title("Rotated Image")

    scaled_img, _ = scale_image(image)
    plt.subplot(132),plt.imshow(scaled_img), plt.title("Scaled Image")

    skewed_img, _ = skew_image(image)
    plt.subplot(133),plt.imshow(skewed_img), plt.title("Skewed Image"), plt.show()

    combined_warp_img, _ = combined_warp(image)
    plt.figure(figsize=(10,5))
    plt.subplot(121),plt.imshow(combined_warp_img), plt.title("Combined Warp Image")

    combined_warp_biliear_img = combined_warp_biinear(image)
    plt.subplot(122),plt.imshow(combined_warp_biliear_img.astype(np.uint8)), plt.title("Combined Warp Image with Bilinear Interpolation"),plt.show()

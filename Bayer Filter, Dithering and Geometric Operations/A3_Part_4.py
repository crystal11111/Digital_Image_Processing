# Import libraries
from skimage import io
from skimage import exposure
import skimage
from skimage.color import rgb2gray
from skimage import transform
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform
from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity
from skimage.transform import warp
from skimage.transform import SimilarityTransform
from skimage.feature import plot_matches

def part4():
    ''' Stitch two images together '''
    image0 = io.imread('im1.jpg', True)
    image1 = io.imread('im2.jpg', True)

    plt.figure(figsize=(8,12))
    plt.subplot(221),plt.imshow(image0,cmap='gray'),plt.title("First Image")
    plt.subplot(222),plt.imshow(image1,cmap='gray'),plt.title("Second Image")

    # -------- Feature detection and matching -----

    # Initiate ORB detector
    orb = ORB(n_keypoints=2000)

    # Find the keypoints and descriptors
    orb.detect_and_extract(image0)
    keypoints0 = orb.keypoints
    descriptors0 = orb.descriptors

    orb.detect_and_extract(image1)
    keypoints1 = orb.keypoints
    descriptors1 = orb.descriptors
    
    # initialize Brute-Force matcher and exclude outliers. See match descriptor function.
    matches = match_descriptors(descriptors0, descriptors1, cross_check=True)

    '''
    # Apply ratio test and convert matches to correct type
    good_matches = matches[matches[:, 0] < 0.75 * matches[:, 1]]

    # Convert keypoints to (x, y) coordinates
    src_pts = np.float32([keypoints0[match[0], :2] for match in good_matches])
    dst_pts = np.float32([keypoints1[match[1], :2] for match in good_matches])
    '''
    src = keypoints1[matches[:, 1]][:, ::-1]
    dst = keypoints0[matches[:, 0]][:, ::-1]

    # -------- Transform estimation -------

    # Compute homography matrix using ransac and ProjectiveTransform
    model_robust, inliers = ransac((src, dst),ProjectiveTransform, min_samples=4, residual_threshold=2, max_trials=2000)

    # ------------- Warping ----------------
    #Next, we produce the panorama itself. The first step is to find the shape of the output image by considering the extents of all warped images.

    r, c = image1.shape[:2]

    # Note that transformations take coordinates in (x, y) format,
    # not (row, column), in order to be consistent with most literature.
    corners = np.array([[0, 0],
                        [0, r],
                        [c, 0],
                        [c, r]])

    # Warp the image corners to their new positions.
    warped_corners = model_robust(corners)

    # Find the extents of both the reference image and
    # the warped target image.
    all_corners = np.vstack((warped_corners, corners))

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)

    output_shape = (corner_max - corner_min)
    output_shape = np.ceil(output_shape[::-1])

    # ----- Note: The images are now warped according to the estimated transformation model.

    # A shift is added to ensure that both images are visible in their entirety. Note that warp takes the inverse mapping as input.
    offset = SimilarityTransform(translation=-corner_min)

    image0_ = warp(image0, offset.inverse,
                output_shape=output_shape)

    image1_ = warp(image1, (model_robust + offset).inverse,
                output_shape=output_shape)

    plt.subplot(223),plt.imshow(image0_, cmap="gray"),plt.title("Warped first image")
    plt.subplot(224),plt.imshow(image1_, cmap="gray"),plt.title("Warped second image")
    plt.show()

    #An alpha channel is added to the warped images before merging them into a single image:

    def add_alpha(image, background=-1):
        """Add an alpha layer to the image.

        The alpha layer is set to 1 for foreground
        and 0 for background.
        """
        rgb = gray2rgb(image)
        alpha = (image != background)
        return np.dstack((rgb, alpha))


    # add alpha to the image0 and image1
    image0_alpha = add_alpha(image0_)
    image1_alpha = add_alpha(image1_)

    # TODO: merge the alpha added image (only change the next line)
    merged = image0_alpha * 0.5 + image1_alpha * 0.5
    alpha = merged[..., 3]
    merged /= np.maximum(alpha, 1)[..., np.newaxis]

    # The summed alpha layers give us an indication of how many images were combined to make up each pixel.  
    # Divide by the number of images to get an average.

    plt.figure(figsize=(10,8))
    plt.imshow(merged, cmap="gray")
    plt.show()
    
    # TODO: randomly select 10 inlier matches and show them using plot_matches
    # Randomly select 10 inlier matches and show them using plot_matches
    num_inliers_to_display = 10
    random_indices = np.random.choice(inliers.sum(), num_inliers_to_display, replace=False)

    image0 = io.imread('im1.jpg')
    image1 = io.imread('im2.jpg')

    # Get the randomly selected inlier matches
    random_inlier_matches = matches[inliers][random_indices]

    # Display the selected inlier matches using plot_matches
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_matches(ax, image0, image1, keypoints0, keypoints1, random_inlier_matches, only_matches=True)
    plt.show()


if __name__ == "__main__":
    part4()

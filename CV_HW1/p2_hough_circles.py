#!/usr/bin/env python3
import cv2
import numpy as np
from scipy.signal import convolve2d

def detect_edges(image):
    """Find edge points in a grayscale image.

    Args:
    - image (2D uint8 array): A grayscale image.

    Return:
    - edge_image (2D float array): A heat map where the intensity at each point
        is proportional to the edge magnitude.
    """
    # TODO
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float64)
    sobel_x = convolve2d(image, kernel, mode='same', boundary='symm')

    kernel = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float64)
    sobel_y = convolve2d(image, kernel, mode='same', boundary='symm')

    edge_image = np.sqrt(sobel_x**2 + sobel_y**2)

    edge_image = (edge_image / edge_image.max()) * 255.0

    return edge_image.astype(np.float32)


def hough_circles(edge_image, edge_thresh, radius_values):
    """Threshold edge image and calculate the Hough transform accumulator array.

    Args:
    - edge_image (2D float array): An H x W heat map where the intensity at each
        point is proportional to the edge magnitude.
    - edge_thresh (float): A threshold on the edge magnitude values.
    - radius_values (1D int array): An array of R possible radius values.

    Return:
    - thresh_edge_image (2D bool array): Thresholded edge image indicating
        whether each pixel is an edge point or not.
    - accum_array (3D int array): Hough transform accumulator array. Should have
        shape R x H x W.
    """
    # TODO
    H, W = edge_image.shape
    R = len(radius_values)

    thresh_edge_image = edge_image > edge_thresh

    accum_array = np.zeros((R, H, W), dtype=np.int64)

    y_idxs, x_idxs = np.where(thresh_edge_image)
    for r_idx, radius in enumerate(radius_values):
        for x, y in zip(x_idxs, y_idxs):
            for theta in np.linspace(0, 2 * np.pi, num=100):
                a = int(x - radius * np.cos(theta))
                b = int(y - radius * np.sin(theta))
                if 0 <= a < W and 0 <= b < H:
                    accum_array[r_idx, b, a] += 1
    
    return thresh_edge_image, accum_array


def find_circles(image, accum_array, radius_values, hough_thresh):
    """Find circles in an image using output from Hough transform.

    Args:
    - image (3D uint8 array): An H x W x 3 BGR color image. Here we use the
        original color image instead of its grayscale version so the circles
        can be drawn in color.
    - accum_array (3D int array): Hough transform accumulator array having shape
        R x H x W.
    - radius_values (1D int array): An array of R radius values.
    - hough_thresh (int): A threshold of votes in the accumulator array.

    Return:
    - circles (list of 3-tuples): A list of circle parameters. Each element
        (r, y, x) represents the radius and the center coordinates of a circle
        found by the program.
    - circle_image (3D uint8 array): A copy of the original image with detected
        circles drawn in color.
    """
    # TODO
    circle_indices = np.where(accum_array > hough_thresh)
    circles = []
    circle_image = image.copy()

    for r_idx, y_idx, x_idx in zip(*circle_indices):
        radius = radius_values[r_idx]
        circles.append((radius, y_idx, x_idx))
        cv2.circle(circle_image, (x_idx, y_idx), radius, (0, 255, 0), 2)

    return circles, circle_image


if __name__ == "__main__":
    # TODO
    print("start")
    image = cv2.imread("data/coins.png", cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edge_image = detect_edges(gray_image)
    cv2.imwrite("output/coins_edges.png", edge_image)

    edge_thresh = 100
    radius_values = np.arange(20, 41)
    thresh_edge_image, accum_array = hough_circles(
        edge_image, edge_thresh, radius_values
    )

    hough_thresh = 50
    circles, circle_image = find_circles(
        image, accum_array, radius_values, hough_thresh
    )
    cv2.imwrite("output/coins_circles.png", circle_image)

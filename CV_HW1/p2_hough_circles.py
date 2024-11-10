#!/usr/bin/env python3
import cv2
import numpy


def detect_edges(image):
    """Find edge points in a grayscale image.

    Args:
    - image (2D uint8 array): A grayscale image.

    Return:
    - edge_image (2D float array): A heat map where the intensity at each point
        is proportional to the edge magnitude.
    """
    # TODO
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # 计算边缘强度（梯度的幅值）
    edge_image = numpy.sqrt(sobel_x**2 + sobel_y**2)

    # 归一化边缘强度图像到0-255范围
    edge_image = (edge_image / edge_image.max()) * 255.0
    return edge_image.astype(numpy.float32)


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

    # 阈值化边缘图像
    thresh_edge_image = edge_image > edge_thresh

    # 创建霍夫累加器数组
    accum_array = numpy.zeros((R, H, W), dtype=numpy.int32)

    # 对边缘点遍历并投票
    y_idxs, x_idxs = numpy.where(thresh_edge_image)
    for r_idx, radius in enumerate(radius_values):
        for x, y in zip(x_idxs, y_idxs):
            for theta in numpy.linspace(0, 2 * numpy.pi, num=100):
                a = int(x - radius * numpy.cos(theta))
                b = int(y - radius * numpy.sin(theta))
                if 0 <= a < W and 0 <= b < H:
                    accum_array[r_idx, b, a] += 1


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
    circles = []
    circle_image = image.copy()

    # 遍历累加器查找高于阈值的圆
    for r_idx, radius in enumerate(radius_values):
        for y in range(accum_array.shape[1]):
            for x in range(accum_array.shape[2]):
                if accum_array[r_idx, y, x] >= hough_thresh:
                    circles.append((radius, y, x))
                    cv2.circle(circle_image, (x, y), radius, (0, 255, 0), 2)

    return circles, circle_image


if __name__ == "__main__":
    # TODO
    print("start")
    image = cv2.imread("data/coins.png", cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测边缘
    edge_image = detect_edges(gray_image)
    cv2.imwrite("output/coins_edges.png", edge_image)

    # 霍夫变换检测圆
    edge_thresh = 100  # 设置边缘阈值
    radius_values = numpy.arange(20, 41)  # 圆的半径范围
    thresh_edge_image, accum_array = hough_circles(
        edge_image, edge_thresh, radius_values
    )

    # 在累加器中查找圆并绘制
    hough_thresh = 150  # 设置霍夫累加器阈值
    circles, circle_image = find_circles(
        image, accum_array, radius_values, hough_thresh
    )
    cv2.imwrite("output/coins_circles.png", circle_image)

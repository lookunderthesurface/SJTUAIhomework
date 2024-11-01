#!/usr/bin/env python3
import cv2
import numpy as np
import sys


def binarize(gray_image, thresh_val):
    # TODO: 255 if intensity >= thresh_val else 0
    rows, cols = gray_image.shape

    binary_image = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if gray_image[i, j] >= thresh_val:
                binary_image[i, j] = 255
    return binary_image


def label(binary_image):
    # TODO
    labeled_image = np.zeros_like(binary_image, dtype=np.int32)
    label = 0

    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] == 255 and labeled_image[i, j] == 0:
                label += 1
                if label == 255:
                    print("label Error!")
                    return
                stack = [(x, y)]
                while stack:
                    x, y = stack.pop()
                    if (
                        x < 0
                        or x >= binary_image.shape[0]
                        or y < 0
                        or y >= binary_image.shape[1]
                        or labeled_image[x, y] != 0
                    ):
                        continue
                    if binary_image[x, y] == 255:
                        labeled_image[x, y] = label
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            stack.append((x + dx, y + dy))
    if label > 0:
        for i in range(binary_image.shape[0]):
            for j in range(binary_image.shape[1]):
                binary_image[i, j] *= 255 / label

    return labeled_image


def get_attribute(labeled_image):
    # TODO
    attribute_list = []
    num_labels = int(np.max(labeled_image))
    
    for label in range(1, num_labels):
        labeled_mask = (labeled_image == label)
        
        y_coords, x_coords = np.where(labeled_mask)
        centroid_y, centroid_x = np.mean(y_coords), np.mean(x_coords)
        position = {'x': float(centroid_x), 'y': float(centroid_y)}
        
        orientation = 0.0
        area = np.sum(labeled_mask)
        perimeter = 0
        for i in range(labeled_mask.shape[0]):
            for j in range(labeled_mask.shape[1]):
                if labeled_mask[i, j] == 1:
                    perimeter += 1
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if 0 <= i + di < labeled_mask.shape[0] and 0 <= j + dj < labeled_mask.shape[1] and labeled_mask[i + di, j + dj] == 1:
                            perimeter -= 1
        roundedness = area / (perimeter ** 2) if perimeter > 0 else 0.0
        
        attribute_list.append({
            'position': position,
            'orientation': float(orientation),
            'roundedness': float(roundedness)
        })
    
    return attribute_list

def main(argv):
    img_name = argv[0]
    thresh_val = int(argv[1])
    img = cv2.imread("data/" + img_name + ".png", cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary_image = binarize(gray_image, thresh_val=thresh_val)
    labeled_image = label(binary_image)
    attribute_list = get_attribute(labeled_image)

    cv2.imwrite("output/" + img_name + "_gray.png", gray_image)
    cv2.imwrite("output/" + img_name + "_binary.png", binary_image)
    cv2.imwrite("output/" + img_name + "_labeled.png", labeled_image)
    print(attribute_list)


if __name__ == "__main__":
    main(sys.argv[1:])  # python3 p1_object_attributes.py two_objects 128

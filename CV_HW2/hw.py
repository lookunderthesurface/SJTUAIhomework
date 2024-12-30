import cv2
import numpy as np
import glob

def read_images(image_directory):
    # Read all jpg images from the specified directory
    return [cv2.imread(image_path) for image_path in glob.glob(f"{image_directory}/*.jpg")]

def find_image_points(images, pattern_size):
    world_points = []
    image_points = []
    
    # TODO: Initialize the chessboard world coordinate points
    def init_world_points(pattern_size):
        # Students should fill in code here to generate the world coordinates of the chessboard
        world_points = []
        for row in range(pattern_size[1]):  # For each row
            for col in range(pattern_size[0]):  # For each column
                # Assuming unit square size of 1 for simplicity
                world_points.append([col, row, 0])
        return np.array(world_points, dtype=np.float32)
    
    # TODO: Detect chessboard corners in each image
    def detect_corners(image, pattern_size):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH)
        return corners.reshape(-1, 2) if found else None
    
    wrong_corner = 0

    # TODO: Complete the loop below to obtain the corners of each image and the corresponding world coordinate points
    for image in images:
        corners = detect_corners(image, pattern_size)
        if corners is not None:
            # Add image corners
            image_points.append(corners)
            # Add the corresponding world points
            world_points.append(init_world_points(pattern_size))
        else: wrong_corner+=1
    print(wrong_corner)
    
    return world_points, image_points

def calibrate_camera(world_points, image_points):
    assert len(world_points) == len(image_points), "The number of world coordinates and image coordinates must match"
    
    num_points = len(world_points)
    A = []
    B = []
    K = np.zeros((4, 4))
    P = None

    # TODO main loop, use least squares to solve for P and then decompose P to get K and R
    # The steps are as follows:
    # 1. Construct the matrix A and B
    # 2. Solve for P using least squares
    # 3. Decompose P to get K and R
    for i in range(num_points):
        X, Y, Z = world_points[i][0]
        print(world_points[i][0])
        u, v = image_points[i][0]
        
        row1 = [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
        row2 = [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]
        
        A.append(row1)
        A.append(row2)
        
        B.append(u)
        B.append(v)
    
    A = np.array(A)
    B = np.array(B)

    P, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    P = P.reshape(3, 4)
    print(P)
    K, R = np.linalg.qr(P[:, :3])

    if np.linalg.det(K) < 0:
        K = -K
    
    return K, P

# Main process
image_path = 'Sample_Calibration_Images'
images = read_images(image_path)

# TODO: I'm too lazy to count the number of chessboard squares, count them yourself
pattern_size = (31, 23)  # The pattern size of the chessboard 

world_points, image_points = find_image_points(images, pattern_size)

camera_matrix, camera_extrinsics = calibrate_camera(world_points, image_points)

print("Camera Calibration Matrix:")
print(camera_matrix)

def test(image_directory, pattern_size):
    # In this function, you are allowed to use OpenCV to verify your results. This function is optional and will not be graded.
    # return None, directly print the results
    # TODO
    print("Camera Calibration Matrix by OpenCV:")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(world_points, image_points, images[0].shape[:2], None, None)

    if ret:
        print("Camera Matrix:\n", camera_matrix)
    else:
        print("Calibration failed.")

    reprojection_error(world_points, image_points, camera_matrix)



def reprojection_error(world_points, image_points, camera_matrix):
    # In this function, you are allowed to use OpenCV to verify your results.
    # show the reprojection error of each image
    total_error = 0
    print(len(world_points[0]))
    for i in range(len(world_points)):
        print(world_points[i].shape)
        Xw = np.vstack((world_points[i], np.ones((len(world_points[i]),1))))
        projected = np.dot(camera_extrinsics, Xw)
        projected = projected[:2] / projected[2]  # Convert from homogeneous coordinates
        
        u, v = image_points[i][0][0], image_points[i][0][1]
        error = np.linalg.norm([u - projected[0], v - projected[1]])  # Euclidean distance error
        total_error += error
    
    return total_error / len(world_points)

print("Camera Calibration Matrix by OpenCV:")
test(image_path, pattern_size)

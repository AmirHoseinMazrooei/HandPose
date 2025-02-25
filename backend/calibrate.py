import cv2
import numpy as np
import cv2.aruco as aruco
import glob

# Load ChArUco board using the factory method
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
squaresX = 4
squaresY = 6
squareLength = 0.03  # In meters
markerLength = 0.02  # In meters

# Correctly create the CharucoBoard using CharucoBoard_create
charuco_board = aruco.CharucoBoard(
    (squaresX,squaresY),
    squareLength=squareLength,
    markerLength=markerLength,
    dictionary=aruco_dict
)

charuco_params = aruco.DetectorParameters()

# Lists to store corners
all_corners = []
all_ids = []
image_size = None

# Load calibration images
image_files = glob.glob("calibration_images/*.png")  # Change to your image path

for image_file in image_files:
    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"Processing {image_file}...")
    # Detect ArUco markers
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=charuco_params)

    if ids is not None:
        # Detect ChArUco board
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=charuco_board
        )

        if charuco_corners is not None and charuco_ids is not None and retval > 3:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)

            # Draw detections
            aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)


    if image_size is None:
        image_size = gray.shape[::-1]
    print(f"Image Size: {image_size}")

cv2.destroyAllWindows()

# Perform camera calibration if enough detections are available
if len(all_corners) > 0 and len(all_ids) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=charuco_board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    if ret:
        # Save calibration parameters
        np.savez("camera_calibration.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

        print("Camera Calibration Successful!")
        print("Camera Matrix:\n", camera_matrix)
        print("Distortion Coefficients:\n", dist_coeffs)
    else:
        print("Camera calibration failed.")
else:
    print("Not enough ChArUco detections for calibration.")
import os
from flask import Flask, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from flask_cors import CORS
import csv
import time

plt.switch_backend('Agg')

# Flask App Setup
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost", "http://127.0.0.1"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)



# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
POSE_LANDMARKS = [landmark.name.lower() for landmark in mp_pose.PoseLandmark]
HAND_LANDMARKS = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
]


def process_pose(rgb_frame):
    return pose.process(rgb_frame)

def process_hands(rgb_frame):
    return hands.process(rgb_frame)

# Helper function to process video
def process_video(input_path, output_video_path, output_csv_path, hand_preference,cm_value=2, px_value=14):
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Camera calibration parameters
    fx = 1000  # Example focal length in pixels (replace with actual calibration)
    fy = 1000  # Example focal length in pixels (replace with actual calibration)
    cx = width / 2  # Principal point x
    cy = height / 2  # Principal point y

    # Camera matrix
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    # Known scale: 2cm = 14px → 1px = 2/14 cm ≈ 0.142857 cm/pixel
    scaling_factor = cm_value / px_value  # cm per pixel

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    csv_data = []
    frame_count = 0
    frame_skip = 1  # Adjust as needed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_results = process_hands(rgb_frame)

        # Draw hand landmarks and extract data
        frame_data = {}
        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                hand_label = handedness.classification[0].label.lower()
                if hand_label != hand_preference.lower():
                    continue  # Skip if not the preferred hand

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract hand data
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    if idx < len(HAND_LANDMARKS):  # Safety check
                        landmark_name = HAND_LANDMARKS[idx]

                        # Convert normalized coordinates to pixel coordinates
                        x_pixel = landmark.x * width
                        y_pixel = landmark.y * height

                        # Apply scaling factor (convert to centimeters)
                        x_cm = (x_pixel - cx) * scaling_factor
                        y_cm = (y_pixel - cy) * scaling_factor
                        z_cm = landmark.z * scaling_factor * fx  # Approximation for z

                        frame_data[f"{hand_label}_{landmark_name}_x"] = x_cm  # in centimeters
                        frame_data[f"{hand_label}_{landmark_name}_y"] = y_cm  # in centimeters
                        frame_data[f"{hand_label}_{landmark_name}_z"] = z_cm  # in centimeters

        csv_data.append(frame_data)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # Save CSV data
    df = pd.DataFrame(csv_data)
    df.to_csv(output_csv_path, index=False)

    # Update calculation functions to use the preferred hand
    calculate_all_thumb_finger_distances(output_csv_path, hand_preference)
    calculate_all_joint_speeds(output_csv_path, hand_preference)
    calculate_all_joint_positions(output_csv_path, hand_preference)
    calculate_all_joint_accelerations(output_csv_path, hand_preference)


# Route to upload and process video
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    video_file = request.files['video']
    hand_preference = request.form.get('hand', 'right')  # Default to 'right' if not provided
    scale_cm = request.form.get('scale_cm', '2')  # Default to 2 cm
    scale_px = request.form.get('scale_px', '14')  # Default to 14 px
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        cm_value = float(scale_cm)
        px_value = float(scale_px)
    except ValueError:
        return jsonify({'error': 'Invalid scale values provided'}), 400
    # Save the uploaded file
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(input_path)

    # Set output paths
    output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], f"annotated_{video_file.filename}")
    output_csv_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{os.path.splitext(video_file.filename)[0]}.csv")

    # Process the video
    try:
        output_video_path, output_csv_path, output_thumb_distance, output_wrist_speed, output_wrist_coordinates, fingertip_position_plot = process_video(
            input_path, output_video_path, output_csv_path, hand_preference,cm_value,px_value
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'annotated_video': f"/outputs/annotated_{video_file.filename}",
        'pose_data_csv': f"/outputs/{os.path.splitext(video_file.filename)[0]}.csv",
        'thumb_distance': output_thumb_distance,
        'wrist_speed': output_wrist_speed,
        'wrist_x': output_wrist_coordinates.get('X'),
        'wrist_y': output_wrist_coordinates.get('Y'),
        'wrist_z': output_wrist_coordinates.get('Z'),
        'fingertips_plot': fingertip_position_plot,
    })

# Route to serve processed files

def SavePlots(title,value_name, values, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(values)), values, color='blue')
    plt.xlabel('Frame')
    plt.ylabel(value_name)
    plt.title(title)
    plt.legend()
    plt.grid(True)
        
        # Save the plot
    plt.savefig(output_path)
    plt.close()

def calculate_all_joint_speeds(csv_path, hand_preference):
    """
    For the given CSV and hand preference, compute the speed for selected joints:
      - wrist,
      - thumb_tip, index_finger_tip, middle_finger_tip, ring_finger_tip, pinky_tip,
      - index_finger_mcp (the proximal joint of index).
    
    Speed for each joint is computed as the Euclidean distance between consecutive frames.
    A CSV is saved for each joint with:
      - 'speed': each value formatted as a float with two decimals,
      - 'timestamp': computed as (frame index + 1)/59.98 formatted with three decimals.
    Missing speed values are filled using the same logic as in the thumb–finger distances method.
    A plot is created for each joint’s speed.
    
    Returns:
      A dictionary mapping each joint key to its CSV and plot output paths.
    """
    import pandas as pd
    import numpy as np
    import os

    output_folder = app.config['OUTPUT_FOLDER']
    data = pd.read_csv(csv_path)

    # Define the target joints. The key is used for naming; the value is the landmark name.
    joints = {
        "wrist": "wrist",
        "thumb_tip": "thumb_tip",
        "index_finger_tip": "index_finger_tip",
        "middle_finger_tip": "middle_finger_tip",
        "ring_finger_tip": "ring_finger_tip",
        "pinky_tip": "pinky_tip",
        "index_finger_mcp": "index_finger_mcp"  # proximal joint of index
    }

    results = {}

    for joint_key, joint in joints.items():
        # Retrieve coordinate lists for this joint
        x = data.get(f"{hand_preference}_{joint}_x", [])
        y = data.get(f"{hand_preference}_{joint}_y", [])
        z = data.get(f"{hand_preference}_{joint}_z", [])
        
        # Compute speed as Euclidean distance between consecutive frames
        speeds = []
        # Note: Speeds will have one fewer value than the total frame count.
        for i in range(1, len(x)):
            speed = np.sqrt((x[i] - x[i - 1])**2 +
                            (y[i] - y[i - 1])**2 +
                            (z[i] - z[i - 1])**2)
            speeds.append(speed)
        
        # Create DataFrame. Note that the timestamp is computed for each speed entry,
        # corresponding to frame index 1 to N-1.
        speed_df = pd.DataFrame({
            'speed': [float(f"{s:.2f}") for s in speeds],
            'timestamp': [float(f"{(i + 1) / 59.98:.3f}") for i in range(1, len(x))]
        })
        
        # Fill any missing speed values using the same logic as before.
        for i in range(len(speed_df)):
            if pd.isna(speed_df.loc[i, 'speed']):
                if i == 0 or i == len(speed_df) - 1:
                    speed_df.loc[i, 'speed'] = -1
                else:
                    prev_val = speed_df.loc[i - 1, 'speed']
                    next_val = speed_df.loc[i + 1, 'speed']
                    if pd.isna(prev_val) or pd.isna(next_val):
                        speed_df.loc[i, 'speed'] = -1
                    else:
                        speed_df.loc[i, 'speed'] = (prev_val + next_val) / 2
        
        # Define output CSV and plot filenames/paths.
        csv_filename = f"{hand_preference}_{joint}_speed.csv"
        csv_output_path = os.path.join(output_folder, csv_filename)
        plot_filename = f"{hand_preference}_{joint}_speed_plot.png"
        plot_output_path = os.path.join(output_folder, plot_filename)
        
        # Save the CSV file.
        speed_df.to_csv(csv_output_path, index=False)
        
        # Save the plot using your existing SavePlots function.
        # (Assuming SavePlots(title, value_name, values, output_path) exists.)
        SavePlots(
            f"{hand_preference.capitalize()} {joint.replace('_', ' ').capitalize()} Speed",
            "Speed",
            speeds,
            plot_output_path
        )
        
        # Record the output paths.
        results[joint_key] = {
            "csv": f"/outputs/{csv_filename}",
            "plot": f"/outputs/{plot_filename}"
        }

    return results

def calculate_all_thumb_finger_distances(csv_path, hand_preference):
    """
    For a given CSV and hand preference, calculate the Euclidean distance between the thumb tip
    and each other finger tip (index, middle, ring, pinky). The CSV for each distance is saved with:
      - 'distance': each value formatted as a float with two decimals,
      - 'timestamp': computed as (frame index + 1)/59.98 formatted with three decimals.
    A plot is saved for each distance and a dictionary of output paths is returned.
    """
    import pandas as pd
    import numpy as np
    import os

    output_folder = app.config['OUTPUT_FOLDER']
    data = pd.read_csv(csv_path)
    
    # Extract thumb coordinates
    thumb_x = data.get(f'{hand_preference}_thumb_tip_x', [])
    thumb_y = data.get(f'{hand_preference}_thumb_tip_y', [])
    thumb_z = data.get(f'{hand_preference}_thumb_tip_z', [])
    
    # Define target fingers with their landmark keys
    finger_landmarks = {
        "index": "index_finger_tip",
        "middle": "middle_finger_tip",
        "ring": "ring_finger_tip",
        "pinky": "pinky_tip"
    }
    
    results = {}
    
    for finger_name, landmark in finger_landmarks.items():
        # Extract target finger coordinates
        finger_x = data.get(f'{hand_preference}_{landmark}_x', [])
        finger_y = data.get(f'{hand_preference}_{landmark}_y', [])
        finger_z = data.get(f'{hand_preference}_{landmark}_z', [])
        
        # Calculate distances for each frame
        distances = []
        for i in range(len(thumb_x)):
            if i >= len(finger_x):
                break
            distance = np.sqrt(
                (thumb_x[i] - finger_x[i]) ** 2 +
                (thumb_y[i] - finger_y[i]) ** 2 +
                (thumb_z[i] - finger_z[i]) ** 2
            )
            distances.append(distance)
        
        # Create DataFrame with formatted distances and timestamps
        distance_df = pd.DataFrame({
            'distance': [float(f"{d:.2f}") for d in distances],
            'timestamp': [float(f"{(i + 1) / 59.98:.3f}") for i in range(len(distances))]
        })
        
        # Fill any missing distance values (using the same logic as before)
        for i in range(len(distance_df)):
            if pd.isna(distance_df.loc[i, 'distance']):
                if i == 0 or i == len(distance_df) - 1:
                    distance_df.loc[i, 'distance'] = -1
                else:
                    prev_val = distance_df.loc[i - 1, 'distance']
                    next_val = distance_df.loc[i + 1, 'distance']
                    if pd.isna(prev_val) or pd.isna(next_val):
                        distance_df.loc[i, 'distance'] = -1
                    else:
                        distance_df.loc[i, 'distance'] = (prev_val + next_val) / 2

        # Define output CSV and plot paths
        csv_filename = f"{hand_preference}_thumb_{finger_name}_distance.csv"
        csv_output_path = os.path.join(output_folder, csv_filename)
        plot_filename = f"{hand_preference}_thumb_{finger_name}_distance_plot.png"
        plot_output_path = os.path.join(output_folder, plot_filename)
        
        # Save the CSV file
        distance_df.to_csv(csv_output_path, index=False)
        
        # Save the plot (using your SavePlots function)
        SavePlots(
            f"{hand_preference.capitalize()} Thumb-{finger_name.capitalize()} Distance", 
            "Distance", 
            distances, 
            plot_output_path
        )
        
        # Store the output paths in results dictionary
        results[finger_name] = {
            "csv": f"/outputs/{csv_filename}",
            "plot": f"/outputs/{plot_filename}"
        }
    
    return results


def calculate_all_joint_positions(csv_path, hand_preference):
    """
    For the given pose CSV and hand preference, extract the positions for the following joints:
      - Wrist,
      - Thumb tip, index finger tip, middle finger tip, ring finger tip, pinky tip,
      - Index finger mcp (the proximal joint of index).
    
    For each joint, a CSV is created with:
      - 'timestamp': (frame index + 1)/59.98 formatted with three decimals,
      - 'x': x coordinate formatted with two decimals,
      - 'y': y coordinate formatted with two decimals,
      - 'z': z coordinate formatted with two decimals.
    Also, a plot is generated (plotting x, y, and z over frame index) using the SavePlots function.
    
    Returns:
      A dictionary mapping each joint key to its CSV and plot output paths.
    """
    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    output_folder = app.config['OUTPUT_FOLDER']
    data = pd.read_csv(csv_path)

    # Define the target joints.
    joints = {
        "wrist": "wrist",
        "thumb_tip": "thumb_tip",
        "index_finger_tip": "index_finger_tip",
        "middle_finger_tip": "middle_finger_tip",
        "ring_finger_tip": "ring_finger_tip",
        "pinky_tip": "pinky_tip",
        "index_finger_mcp": "index_finger_mcp"  # proximal joint of index
    }

    results = {}

    for joint_key, joint in joints.items():
        # Extract the coordinate lists for this joint.
        x_coords = data.get(f"{hand_preference}_{joint}_x", [])
        y_coords = data.get(f"{hand_preference}_{joint}_y", [])
        z_coords = data.get(f"{hand_preference}_{joint}_z", [])

        # Create a CSV file with the timestamp and positions.
        # Timestamp is computed as (frame index + 1) / 59.98 and formatted to three decimals.
        timestamps = [float(f"{(i + 1) / 59.98:.3f}") for i in range(len(x_coords))]
        # Format positions to two decimals.
        x_formatted = [float(f"{val:.2f}") for val in x_coords]
        y_formatted = [float(f"{val:.2f}") for val in y_coords]
        z_formatted = [float(f"{val:.2f}") for val in z_coords]

        pos_df = pd.DataFrame({
            'timestamp': timestamps,
            'x': x_formatted,
            'y': y_formatted,
            'z': z_formatted
        })

        csv_filename = f"{hand_preference}_{joint}_position.csv"
        csv_output_path = os.path.join(output_folder, csv_filename)
        pos_df.to_csv(csv_output_path, index=False)

        # Create a plot of positions over frame-index.
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(x_coords)), x_coords, marker='o', linestyle='-', label='X', color='red')
        plt.plot(range(len(y_coords)), y_coords, marker='o', linestyle='-', label='Y', color='blue')
        plt.plot(range(len(z_coords)), z_coords, marker='o', linestyle='-', label='Z', color='green')
        plt.xlabel('Frame')
        plt.ylabel('Position (cm)')
        plt.title(f"{hand_preference.capitalize()} {joint.replace('_', ' ').capitalize()} Position")
        plt.legend()
        plt.grid(True)

        plot_filename = f"{hand_preference}_{joint}_position_plot.png"
        plot_output_path = os.path.join(output_folder, plot_filename)
        plt.savefig(plot_output_path)
        plt.close()

        results[joint_key] = {
            "csv": f"/outputs/{csv_filename}",
            "plot": f"/outputs/{plot_filename}"
        }

    return results

def calculate_all_joint_accelerations(csv_path, hand_preference):
    """
    For the given pose CSV and hand preference, compute the acceleration (i.e. change in speed)
    for selected joints:
      - Wrist,
      - Thumb tip, index finger tip, middle finger tip, ring finger tip, pinky tip,
      - Index finger mcp (the proximal joint of index).

    The method first computes the speed (Euclidean distance between positions of consecutive frames)
    and then computes the acceleration as:
         acceleration = (speed[i] - speed[i-1]) / dt,
    where dt = 1/59.98 seconds.

    A CSV is saved for each joint with:
      - 'acceleration': each value formatted as a float with two decimals,
      - 'timestamp': computed as (frame index + 2)/59.98 formatted with three decimals.
    Missing acceleration values are filled using the same logic as before.
    A plot is also saved for each joint’s acceleration.

    Returns:
       A dictionary mapping each joint key to its CSV and plot output paths.
    """
    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    output_folder = app.config['OUTPUT_FOLDER']
    data = pd.read_csv(csv_path)

    # Define the target joints.
    joints = {
        "wrist": "wrist",
        "thumb_tip": "thumb_tip",
        "index_finger_tip": "index_finger_tip",
        "middle_finger_tip": "middle_finger_tip",
        "ring_finger_tip": "ring_finger_tip",
        "pinky_tip": "pinky_tip",
        "index_finger_mcp": "index_finger_mcp"  # proximal joint of index
    }

    results = {}
    dt = 1 / 59.98  # Time difference between frames

    for joint_key, joint in joints.items():
        # Retrieve coordinate lists for the current joint.
        x = data.get(f"{hand_preference}_{joint}_x", [])
        y = data.get(f"{hand_preference}_{joint}_y", [])
        z = data.get(f"{hand_preference}_{joint}_z", [])

        # Compute speeds as the Euclidean distance between consecutive frames.
        speeds = []
        for i in range(1, len(x)):
            speed = np.sqrt((x[i] - x[i - 1])**2 +
                            (y[i] - y[i - 1])**2 +
                            (z[i] - z[i - 1])**2)
            speeds.append(speed)

        # Compute accelerations as the difference between consecutive speeds divided by dt.
        # Thus, acceleration[i] corresponds to the acceleration at frame i+1 (starting at the second speed).
        accelerations = []
        for i in range(1, len(speeds)):
            acc = (speeds[i] - speeds[i - 1]) / dt
            accelerations.append(acc)

        # Create a DataFrame with formatted accelerations and timestamps.
        # Since accelerations are computed for frames 2..N (using positions from frames 1..N),
        # we define timestamp = (i + 2) / 59.98 for each acceleration (i starting at 0).
        acc_timestamps = [float(f"{(i + 2) / 59.98:.3f}") for i in range(len(accelerations))]
        acc_df = pd.DataFrame({
            'acceleration': [float(f"{a:.2f}") for a in accelerations],
            'timestamp': acc_timestamps
        })

        # Fill any missing acceleration values using the same interpolation logic.
        for i in range(len(acc_df)):
            if pd.isna(acc_df.loc[i, 'acceleration']):
                if i == 0 or i == len(acc_df) - 1:
                    acc_df.loc[i, 'acceleration'] = -1
                else:
                    prev_val = acc_df.loc[i - 1, 'acceleration']
                    next_val = acc_df.loc[i + 1, 'acceleration']
                    if pd.isna(prev_val) or pd.isna(next_val):
                        acc_df.loc[i, 'acceleration'] = -1
                    else:
                        acc_df.loc[i, 'acceleration'] = (prev_val + next_val) / 2

        # Define output CSV and plot filenames/paths.
        csv_filename = f"{hand_preference}_{joint}_acceleration.csv"
        csv_output_path = os.path.join(output_folder, csv_filename)
        plot_filename = f"{hand_preference}_{joint}_acceleration_plot.png"
        plot_output_path = os.path.join(output_folder, plot_filename)

        # Save the acceleration CSV.
        acc_df.to_csv(csv_output_path, index=False)

        # Create a plot of acceleration over frames.
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(accelerations)), accelerations, marker='o', linestyle='-', color='purple', label='Acceleration')
        plt.xlabel('Frame')
        plt.ylabel('Acceleration (cm/s²)')
        plt.title(f"{hand_preference.capitalize()} {joint.replace('_', ' ').capitalize()} Acceleration")
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_output_path)
        plt.close()

        # Store the output paths.
        results[joint_key] = {
            "csv": f"/outputs/{csv_filename}",
            "plot": f"/outputs/{plot_filename}"
        }

    return results

def calculate_all_joint_accelerations(csv_path, hand_preference):
    """
    For the given pose CSV and hand preference, compute the acceleration (i.e. change in speed)
    for selected joints:
      - Wrist,
      - Thumb tip, index finger tip, middle finger tip, ring finger tip, pinky tip,
      - Index finger mcp (the proximal joint of index).

    The method first computes the speed (Euclidean distance between positions of consecutive frames)
    and then computes the acceleration as:
         acceleration = (speed[i] - speed[i-1]) / dt,
    where dt = 1/59.98 seconds.

    A CSV is saved for each joint with:
      - 'acceleration': each value formatted as a float with two decimals,
      - 'timestamp': computed as (frame index + 2)/59.98 formatted with three decimals.
    Missing acceleration values are filled using the same logic as before.
    A plot is also saved for each joint’s acceleration.

    Returns:
       A dictionary mapping each joint key to its CSV and plot output paths.
    """
    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    output_folder = app.config['OUTPUT_FOLDER']
    data = pd.read_csv(csv_path)

    # Define the target joints.
    joints = {
        "wrist": "wrist",
        "thumb_tip": "thumb_tip",
        "index_finger_tip": "index_finger_tip",
        "middle_finger_tip": "middle_finger_tip",
        "ring_finger_tip": "ring_finger_tip",
        "pinky_tip": "pinky_tip",
        "index_finger_mcp": "index_finger_mcp"  # proximal joint of index
    }

    results = {}
    dt = 1 / 59.98  # Time difference between frames

    for joint_key, joint in joints.items():
        # Retrieve coordinate lists for the current joint.
        x = data.get(f"{hand_preference}_{joint}_x", [])
        y = data.get(f"{hand_preference}_{joint}_y", [])
        z = data.get(f"{hand_preference}_{joint}_z", [])

        # Compute speeds as the Euclidean distance between consecutive frames.
        speeds = []
        for i in range(1, len(x)):
            speed = np.sqrt((x[i] - x[i - 1])**2 +
                            (y[i] - y[i - 1])**2 +
                            (z[i] - z[i - 1])**2)
            speeds.append(speed)

        # Compute accelerations as the difference between consecutive speeds divided by dt.
        # Thus, acceleration[i] corresponds to the acceleration at frame i+1 (starting at the second speed).
        accelerations = []
        for i in range(1, len(speeds)):
            acc = (speeds[i] - speeds[i - 1]) / dt
            accelerations.append(acc)

        # Create a DataFrame with formatted accelerations and timestamps.
        # Since accelerations are computed for frames 2..N (using positions from frames 1..N),
        # we define timestamp = (i + 2) / 59.98 for each acceleration (i starting at 0).
        acc_timestamps = [float(f"{(i + 2) / 59.98:.3f}") for i in range(len(accelerations))]
        acc_df = pd.DataFrame({
            'acceleration': [float(f"{a:.2f}") for a in accelerations],
            'timestamp': acc_timestamps
        })

        # Fill any missing acceleration values using the same interpolation logic.
        for i in range(len(acc_df)):
            if pd.isna(acc_df.loc[i, 'acceleration']):
                if i == 0 or i == len(acc_df) - 1:
                    acc_df.loc[i, 'acceleration'] = -1
                else:
                    prev_val = acc_df.loc[i - 1, 'acceleration']
                    next_val = acc_df.loc[i + 1, 'acceleration']
                    if pd.isna(prev_val) or pd.isna(next_val):
                        acc_df.loc[i, 'acceleration'] = -1
                    else:
                        acc_df.loc[i, 'acceleration'] = (prev_val + next_val) / 2

        # Define output CSV and plot filenames/paths.
        csv_filename = f"{hand_preference}_{joint}_acceleration.csv"
        csv_output_path = os.path.join(output_folder, csv_filename)
        plot_filename = f"{hand_preference}_{joint}_acceleration_plot.png"
        plot_output_path = os.path.join(output_folder, plot_filename)

        # Save the acceleration CSV.
        acc_df.to_csv(csv_output_path, index=False)

        # Create a plot of acceleration over frames.
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(accelerations)), accelerations, marker='o', linestyle='-', color='purple', label='Acceleration')
        plt.xlabel('Frame')
        plt.ylabel('Acceleration (cm/s²)')
        plt.title(f"{hand_preference.capitalize()} {joint.replace('_', ' ').capitalize()} Acceleration")
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_output_path)
        plt.close()

        # Store the output paths.
        results[joint_key] = {
            "csv": f"/outputs/{csv_filename}",
            "plot": f"/outputs/{plot_filename}"
        }

    return results

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5200, debug=True)
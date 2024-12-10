import os
from flask import Flask, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor

# Flask App Setup
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()
hands = mp_hands.Hands()
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
def process_video(input_path, output_video_path, output_csv_path):
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    csv_data = []

    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break


            # Resize frame for faster processing
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process pose and hands in parallel
            with ThreadPoolExecutor() as executor:
                pose_future = executor.submit(process_pose, rgb_frame)
                hands_future = executor.submit(process_hands, rgb_frame)

                pose_results = pose_future.result()
                hands_results = hands_future.result()

            # Draw pose landmarks
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Draw hand landmarks
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract pose and hand data
            frame_data = {}

            # For pose landmarks
            if pose_results.pose_landmarks:
                for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    if idx < len(POSE_LANDMARKS):  # Safety check
                        landmark_name = POSE_LANDMARKS[idx]
                        frame_data[f"{landmark_name}_x"] = landmark.x
                        frame_data[f"{landmark_name}_y"] = landmark.y
                        frame_data[f"{landmark_name}_z"] = landmark.z
                        frame_data[f"{landmark_name}_visibility"] = landmark.visibility

            # For hand landmarks
            if hands_results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                    hand_label = "left" if hand_idx == 0 else "right"
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        if idx < len(HAND_LANDMARKS):  # Safety check
                            landmark_name = HAND_LANDMARKS[idx]
                            frame_data[f"{hand_label}_{landmark_name}_x"] = landmark.x
                            frame_data[f"{hand_label}_{landmark_name}_y"] = landmark.y
                            frame_data[f"{hand_label}_{landmark_name}_z"] = landmark.z

            csv_data.append(frame_data)

            # Write the annotated frame to the output video
            out.write(frame)

            pbar.update(1)

    cap.release()
    out.release()

    # Save CSV data
    pd.DataFrame(csv_data).to_csv(output_csv_path, index=False)
    output_thumb_distance=calculate_finger_thumb_distance(output_csv_path)
    output_wrist_speed=calculate_wrist_speed(output_csv_path)
    output_wrist_coordinates=plot_wrist_individual_coordinates(output_csv_path)
    return output_video_path, output_csv_path, output_thumb_distance, output_wrist_speed, output_wrist_coordinates

# Route to upload and process video
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(input_path)

    # Set output paths
    output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], f"annotated_{video_file.filename}")
    output_csv_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{os.path.splitext(video_file.filename)[0]}.csv")

    # Process the video
    try:
        output_video_path, output_csv_path, output_thumb_distance, output_wrist_speed, output_wrist_coordinates = process_video(input_path, output_video_path, output_csv_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'annotated_video': f"/outputs/annotated_{video_file.filename}",
        'pose_data_csv': f"/outputs/{os.path.splitext(video_file.filename)[0]}.csv",
        'thumb_distance':output_thumb_distance,
        'wrist_speed': output_wrist_speed,
        'wrist_x':output_wrist_coordinates['X'],
        'wrist_y':output_wrist_coordinates['Y'],
        'wrist_z':output_wrist_coordinates['Z'],
    })

# Route to serve processed files
@app.route('/outputs/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

def plot_wrist_individual_coordinates(csv_path):
    # Load data from CSV
    output_folder=app.config['OUTPUT_FOLDER']
    data = pd.read_csv(csv_path)
    wrist_x = data.get('pose_0_x', [])
    wrist_y = data.get('pose_0_y', [])
    wrist_z = data.get('pose_0_z', [])
    
    # Coordinate data dictionary
    coords = {
        'X': wrist_x,
        'Y': wrist_y,
        'Z': wrist_z
    }
    
    output_paths = {}
    for coord, values in coords.items():
        output_path = os.path.join(output_folder, f'wrist_{coord.lower()}_plot.png')
        
        # Plot the data
        SavePlots(f'Wrist {coord}-Position Over Frames',f'Wrist {coord}-Values', values, output_path)
        
        # Store the output path
        output_paths[coord] = output_path
    
    return output_paths

def SavePlots(title,lable, values, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(values)), values, label=lable, color='blue')
    plt.xlabel('Frame')
    plt.ylabel('Position')
    plt.title(title)
    plt.legend()
    plt.grid(True)
        
        # Save the plot
    plt.savefig(output_path)
    plt.close()

def calculate_wrist_speed(csv_path):
    # Load data from CSV
    output_folder=app.config['OUTPUT_FOLDER']
    output_path = os.path.join(output_folder, f'wrist_speed_plot.png')
    data = pd.read_csv(csv_path)
    wrist_x = data.get('left_wrist_x', [])
    wrist_y = data.get('left_wrsit_y', [])
    wrist_z = data.get('left_wrsit_z', [])
    
    # Calculate wrist speed (3D distance between consecutive frames)
    speeds = []
    for i in range(1, len(wrist_x)):
        speed = np.sqrt(
            (wrist_x[i] - wrist_x[i-1])**2 +
            (wrist_y[i] - wrist_y[i-1])**2 +
            (wrist_z[i] - wrist_z[i-1])**2
        )
        speeds.append(speed)
    SavePlots("Wrsit Speed","wrist speed",speeds,output_path)
    return output_path


def calculate_finger_thumb_distance(csv_path):
    # Load data from CSV
    output_folder=app.config['OUTPUT_FOLDER']
    output_path = os.path.join(output_folder, f'thumb_index_distance_plot.png')
    data = pd.read_csv(csv_path)
    thumb_x = data.get('right_thumb_tip_x', [])
    thumb_y = data.get('right_thumb_tip_y', [])
    thumb_z = data.get('right_thumb_tip_z', [])
    index_x = data.get('right_index_finger_tip_x', [])
    index_y = data.get('right_index_finger_tip_y', [])
    index_z = data.get('right_index_finger_tip_z', [])
    
    # Calculate distances
    distances = []
    for i in range(len(thumb_x)):
        distance = np.sqrt(
            (thumb_x[i] - index_x[i])**2 +
            (thumb_y[i] - index_y[i])**2 +
            (thumb_z[i] - index_z[i])**2
        )
        distances.append(distance)
    SavePlots("Thumb To Index Finger Distance","distance",distances,output_path)
    return output_path



@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
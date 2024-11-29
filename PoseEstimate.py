import cv2
import mediapipe as mp
import pandas as pd
import argparse
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
hands = mp_hands.Hands()

# Argument parser setup
parser = argparse.ArgumentParser(description="Perform pose estimation on a video and save annotated video and CSV.")
parser.add_argument("--input", required=True, help="Path to the input AVI video file.")
parser.add_argument("--output_video", required=True, help="Path to save the annotated AVI video.")
parser.add_argument("--output_csv", required=True, help="Path to save the CSV file with pose data.")
parser.add_argument("--resize_width", type=int, default=640, help="Resize width for faster processing.")
parser.add_argument("--frame_skip", type=int, default=1, help="Process every nth frame for faster processing.")
args = parser.parse_args()

# Define landmark names
POSE_LANDMARKS = [landmark.name.lower() for landmark in mp_pose.PoseLandmark]
HAND_LANDMARKS = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
]

# Open the video
cap = cv2.VideoCapture(args.input)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
resize_height = int(height * args.resize_width / width)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
out = cv2.VideoWriter(args.output_video, fourcc, fps, (args.resize_width, resize_height))

# CSV data structure
csv_data = []
frame_index = 0

def process_pose(rgb_frame):
    return pose.process(rgb_frame)

def process_hands(rgb_frame):
    return hands.process(rgb_frame)

# Process video with a progress bar
with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames for faster processing
        if frame_index % args.frame_skip != 0:
            frame_index += 1
            pbar.update(1)
            continue

        # Resize frame for faster processing
        frame = cv2.resize(frame, (args.resize_width, resize_height))
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
        frame_data = {"frame": frame_index}

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

        # Update progress bar
        frame_index += 1
        pbar.update(1)

# Release resources
cap.release()
out.release()
pose.close()
hands.close()

# Save CSV file
csv_df = pd.DataFrame(csv_data)
csv_df.to_csv(args.output_csv, index=False)

print(f"Annotated video saved to {args.output_video}")
print(f"Pose data saved to {args.output_csv}")

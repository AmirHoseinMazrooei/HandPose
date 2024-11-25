import cv2
import mediapipe as mp
import pandas as pd
import argparse
from tqdm import tqdm  # Progress bar library

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
args = parser.parse_args()

# Open the video
cap = cv2.VideoCapture(args.input)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

# CSV data structure
csv_data = []
frame_index = 0

# Process video with a progress bar
with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process pose and hands
        pose_results = pose.process(rgb_frame)
        hands_results = hands.process(rgb_frame)

        # Draw pose landmarks
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Draw hand landmarks
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract pose and hand data
        frame_data = {"frame": frame_index}

        if pose_results.pose_landmarks:
            for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                frame_data[f"pose_{idx}_x"] = landmark.x
                frame_data[f"pose_{idx}_y"] = landmark.y
                frame_data[f"pose_{idx}_z"] = landmark.z

        if hands_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    frame_data[f"hand{hand_idx}_{idx}_x"] = landmark.x
                    frame_data[f"hand{hand_idx}_{idx}_y"] = landmark.y
                    frame_data[f"hand{hand_idx}_{idx}_z"] = landmark.z

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

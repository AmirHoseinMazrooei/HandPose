import cv2
import os

def video_to_frames(video_path, output_folder, start_time=40, end_time=45):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Capture the video
    video_capture = cv2.VideoCapture(video_path)
    
    # Check if the video was successfully loaded
    if not video_capture.isOpened():
        print("Error: Cannot open video file.")
        return
    
    # Get the video's frames per second (FPS) and total frames
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    
    # Calculate frame numbers for the start and end times
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Set the starting frame position
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    while frame_count < end_frame:
        # Read one frame
        ret, frame = video_capture.read()
        
        # Break the loop if no frame is returned (end of video)
        if not ret:
            break
        
        # Save the frame as a JPEG file
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1
    
    # Release the video capture object
    video_capture.release()
    print(f"Extracted frames from {start_time}s to {end_time}s to '{output_folder}'.")

# Usage example
video_path = "P01_Control_NV_Side_Cropped.mp4.re-encoded.1280px.8Mb.avi"  # Replace with your video file
output_folder = "frames"       # Replace with your desired output folder
video_to_frames(video_path, output_folder, start_time=40, end_time=45)

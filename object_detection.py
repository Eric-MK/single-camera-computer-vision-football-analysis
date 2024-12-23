import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO('models/object.pt') 

# Video file path
video_path = 'input_video/08fd33_4.mp4'
output_path = 'output_video/annotated_match.mp4'

# Video capture and output setup
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Batch processing parameters
batch_size = 16
frames = []
frame_indices = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Add frame to batch
    frames.append(frame)
    frame_indices.append(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Optional: Track frame indices

    # Process batch
    if len(frames) == batch_size:
        # Convert frames to a batch of images
        batch = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]  # Convert BGR to RGB for YOLO
        results = model(batch)  # Run inference on batch

        # Annotate and write results
        for result, frame in zip(results, frames):
            annotated_frame = result.plot()  # Use `plot()` for visualization
            out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))  # Convert back to BGR and save
        frames.clear()  # Clear batch

# Process remaining frames in the last batch
if frames:
    batch = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    results = model(batch)
    for result, frame in zip(results, frames):
        annotated_frame = result.plot()
        out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Annotated video saved to {output_path}")


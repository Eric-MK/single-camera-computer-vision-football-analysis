import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv

def process_yolo_video(model_path, video_path, output_path):
    model = YOLO(model_path)
    tracker = sv.ByteTrack(lost_track_buffer=30)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Define colors for each class (BGR format)
    colors = {
        0: (0, 0, 255),  # Ball: Red
        1: (0, 255, 0),  # Goalkeeper: Green
        2: (255, 0, 0),  # Player: Blue
        3: (0, 255, 255),  # Referee: Yellow
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame for YOLO and run inference
        batch = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)]
        results = model(batch)

        # Debug: Check YOLO detections
        result = results[0]
        print(f"Detections: {len(result.boxes)}")
        
        # Build detections for tracker
        bboxes, confidences, class_ids = [], [], []
        for detection in result.boxes:
            class_id = int(detection.cls[0])
            confidence = detection.conf[0]

            # Debug: Print detection info
            print(f"Class ID: {class_id}, Confidence: {confidence}")

            # Use a lower threshold for the ball (class_id == 0 in this example)
            if (class_id == 0 and confidence >= 0.25) or confidence >= 0.3:
                x1, y1, x2, y2 = detection.xyxy[0]
                bboxes.append([x1, y1, x2, y2])
                confidences.append(confidence)
                class_ids.append(class_id)

        # Create detection object
        detections = sv.Detections(xyxy=np.array(bboxes), confidence=np.array(confidences), class_id=np.array(class_ids))
        print(f"Detections ready for tracker: {detections.xyxy.shape[0]}")

        # Update tracker
        tracks = tracker.update_with_detections(detections)
        print(f"Tracks: {len(tracks)}")

        # Annotate frame with different colors for each class
        for track in tracks:
            # Debug: Print track to see its structure
           # print(f"Track: {track}")
            
            bbox = track[0]  # [x1, y1, x2, y2]
            track_id = track[4]  # Assuming track[4] contains the track ID

            # Extract class ID from track[3] (this is where the class ID is stored)
            class_id = int(track[3])  # Class ID is at index 3

            # Get the color based on class_id
            color = colors.get(class_id, (255, 255, 255))  # Default to white if class_id not in colors

            # Draw bounding box
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

            # Label the tracked object
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write annotated frame
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_yolo_video(
        model_path='models/object.pt',
        video_path='input_video/08fd33_4.mp4',
        output_path='output_video/debug_tracked2.mp4'
    )

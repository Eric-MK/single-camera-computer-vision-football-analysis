import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO
import supervision as sv

def get_dominant_color(image, bbox):
    """Extract the dominant color from a bounding box region in the image."""
    x1, y1, x2, y2 = bbox
    player_crop = image[int(y1):int(y2), int(x1):int(x2)]
    # Convert the cropped region to HSV
    hsv_image = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)
    
    # Reshape the image into a list of pixels
    pixels = hsv_image.reshape((-1, 3))
    
    # Apply KMeans clustering to find the dominant color
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(pixels)
    
    # Return the dominant color (centroid of the cluster)
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color

def process_yolo_video_with_teams(model_path, video_path, output_path, num_teams=2):
    model = YOLO(model_path)
    tracker = sv.ByteTrack(lost_track_buffer=30)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    team_colors = []
    all_dominant_colors = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame for YOLO and run inference
        batch = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)]
        results = model(batch)

        # Get the results
        result = results[0]

        # Build detections for tracker
        bboxes, confidences, class_ids = [], [], []
        for detection in result.boxes:
            class_id = int(detection.cls[0])
            confidence = detection.conf[0]

            if (class_id == 0 and confidence >= 0.25) or confidence >= 0.3:
                x1, y1, x2, y2 = detection.xyxy[0]
                bboxes.append([x1, y1, x2, y2])
                confidences.append(confidence)
                class_ids.append(class_id)

        # Create detection object
        detections = sv.Detections(xyxy=np.array(bboxes), confidence=np.array(confidences), class_id=np.array(class_ids))

        # Update tracker
        tracks = tracker.update_with_detections(detections)

        # Collect dominant colors for clustering
        for track in tracks:
            bbox = track[0]  # [x1, y1, x2, y2]
            # Get the dominant color of the player's jersey
            dominant_color = get_dominant_color(frame, bbox)
            all_dominant_colors.append(dominant_color)

        # Perform K-means clustering on the collected dominant colors
        if len(all_dominant_colors) > 0:
            kmeans = KMeans(n_clusters=num_teams)
            kmeans.fit(all_dominant_colors)
            team_colors = kmeans.cluster_centers_

        # Annotate frame with team colors
        for track in tracks:
            bbox = track[0]  # [x1, y1, x2, y2]
            # Get the dominant color of the player's jersey
            dominant_color = get_dominant_color(frame, bbox)
            
            # Find the closest cluster (team) for this color
            distances = np.linalg.norm(team_colors - dominant_color, axis=1)
            team_id = np.argmin(distances)
            
            # Assign a color for the team and annotate
            team_color = team_colors[team_id]
            team_color = tuple(int(c) for c in team_color)  # Convert to int (BGR format)

            label = f"Team {team_id + 1}"  # Label the player by team number
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), team_color, 2)
            cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)

        # Write annotated frame
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_yolo_video_with_teams(
        model_path='models/object.pt',
        video_path='input_video/08fd33_4.mp4',
        output_path='output_video/teams_tracked.mp4',
        num_teams=2  # Adjust this to the number of teams (clusters) you want to differentiate
    )

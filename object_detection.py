import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv
from teams import group_players_by_color, assign_goalkeeper_to_team, get_dominant_color

def process_yolo_video(model_path, video_path, output_path):
    model = YOLO(model_path)
    tracker = sv.ByteTrack(lost_track_buffer=30)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    player_colors = []
    player_bboxes = []
    player_class_ids = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO Inference
        batch = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)]
        results = model(batch)
        result = results[0]

        bboxes, confidences, class_ids = [], [], []

        for detection in result.boxes:
            class_id = int(detection.cls[0])
            confidence = detection.conf[0]

            if confidence >= 0.3:
                x1, y1, x2, y2 = detection.xyxy[0]
                bboxes.append([x1, y1, x2, y2])
                confidences.append(confidence)
                class_ids.append(class_id)

                # Extract jersey colors for players and goalkeepers
                if class_id == 2:  # Players
                    color = get_dominant_color(frame, [x1, y1, x2, y2])
                    player_colors.append(color)
                    player_bboxes.append([x1, y1, x2, y2])
                    player_class_ids.append(class_id)

                if class_id == 1:  # Goalkeeper
                    goalkeeper_color = get_dominant_color(frame, [x1, y1, x2, y2])
                    goalkeeper_bbox = [x1, y1, x2, y2]

        # Group players into teams
        if player_colors:  # Ensure at least some players are detected
            team_labels, team_centers = group_players_by_color(player_colors)
            goalkeeper_team = assign_goalkeeper_to_team(goalkeeper_color, team_centers)

            # Annotate frame
            for i, bbox in enumerate(player_bboxes):
                x1, y1, x2, y2 = map(int, bbox)
                team = team_labels[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Team {team}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Annotate goalkeeper
            x1, y1, x2, y2 = map(int, goalkeeper_bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"Goalkeeper Team {goalkeeper_team}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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

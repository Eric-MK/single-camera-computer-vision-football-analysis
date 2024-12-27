from object_detection import process_yolo_video



if __name__ == "__main__":
    process_yolo_video(
        model_path='models/object.pt',
        video_path='input_video/08fd33_4.mp4',
        output_path='output_video/tracked2.mp4'
    )
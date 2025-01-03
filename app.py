from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import subprocess
from flask_cors import CORS
from team_functions import Club
from object_detection import process_yolo_video_with_teams  

app = Flask(__name__)
CORS(app)

# Define upload and output folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_video'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/process-video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video = request.files['video']
    data = request.form

    if not allowed_file(video.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    # Save the uploaded file
    filename = secure_filename(video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)

    # Extract and validate team colors
    try:
        club1_colors = {
            'player': tuple(map(int, data.get('club1_player_color').split(','))),
            'goalkeeper': tuple(map(int, data.get('club1_goalkeeper_color').split(',')))
        }
        club2_colors = {
            'player': tuple(map(int, data.get('club2_player_color').split(','))),
            'goalkeeper': tuple(map(int, data.get('club2_goalkeeper_color').split(',')))
        }
    except Exception as e:
        return jsonify({'error': f'Invalid color format: {str(e)}'}), 400

    # Define clubs
    club1 = Club(name='Team1', player_jersey_color=club1_colors['player'], goalkeeper_jersey_color=club1_colors['goalkeeper'])
    club2 = Club(name='Team2', player_jersey_color=club2_colors['player'], goalkeeper_jersey_color=club2_colors['goalkeeper'])

    # Output file path
    output_filename = f"processed_{filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    # Process the video
    try:
        # Process the video with YOLO or any other processing you need
        process_yolo_video_with_teams(
            model_path='models/object.pt',  # Update with actual model path
            video_path=video_path,
            output_path=output_path,
            club1=club1,
            club2=club2
        )

        # Convert the processed video to MP4 (H.264 + AAC codec) for browser compatibility
        converted_output_filename = f"converted_{output_filename.split('.')[0]}.mp4"
        converted_output_path = os.path.join(app.config['OUTPUT_FOLDER'], converted_output_filename)
        
        ffmpeg_command = [
            'ffmpeg', '-i', output_path, 
            '-vcodec', 'libx264', '-acodec', 'aac', 
            '-strict', 'experimental', '-b:v', '1000k',
            '-preset', 'fast', '-movflags', '+faststart',  # Optimize for web playback
            converted_output_path
        ]
        subprocess.run(ffmpeg_command, check=True)

    except Exception as e:
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500

    # Return the path to the converted output video
    return jsonify({'message': 'Video processed and converted successfully', 'output_video': f'/output_video/{converted_output_filename}'}), 200

@app.route('/output_video/<filename>', methods=['GET'])
def get_output_video(filename):
    """Serve the processed video to the frontend."""
    # Ensure the mimetype is 'video/mp4' when serving the converted video
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True)

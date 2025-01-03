from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
import os
import subprocess
from flask_cors import CORS
from team_functions import Club
from object_detection import process_yolo_video_with_teams  

# Flask app and database initialization
app = Flask(__name__)
CORS(app)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root@localhost/football_cv'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable track modifications to avoid overhead
app.config['JWT_SECRET_KEY'] = 'your-secret-key'  # Change this to a secure secret
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)  # Token expires in 1 hour

# Initialize the database and JWT manager
db = SQLAlchemy(app)
jwt = JWTManager(app)

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

# User model
class User(db.Model):
    __tablename__ = 'users'  # Specify the name of the table in the database
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)  # Storing hashed password
    created_at = db.Column(db.DateTime, nullable=False, default=db.func.now())

    def __repr__(self):
        return f'<User {self.name}>'

# Route to register a new user
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    # Check if email already exists in the database
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'User with this email already exists'}), 400

    # Hash the password
    password_hash = generate_password_hash(password)

    # Create new user and add to the database
    new_user = User(name=name, email=email, password_hash=password_hash)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully'}), 201

# Route to login an existing user
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    # Retrieve the user by email
    user = User.query.filter_by(email=email).first()
    if user and check_password_hash(user.password_hash, password):
        # User exists and password is correct
        access_token = create_access_token(identity=user.id)  # Use the user's ID as identity
        return jsonify({'access_token': access_token}), 200
    else:
        return jsonify({'error': 'Invalid email or password'}), 401

# Route to process the video (protected by JWT)
@app.route('/process-video', methods=['POST'])
@jwt_required()  # Protect the route with JWT authentication
def process_video():
    # Get the identity (user ID) from the JWT token
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)

    # Log or use the username if needed (e.g., to associate the video with a user)
    print(f"User {user.name} (ID: {current_user_id}) is uploading a video")

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

# Route to serve processed videos
@app.route('/output_video/<filename>', methods=['GET'])
def get_output_video(filename):
    """Serve the processed video to the frontend."""
    # Ensure the mimetype is 'video/mp4' when serving the converted video
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, mimetype='video/mp4')

# Create the database tables (run once to set up the database)
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)

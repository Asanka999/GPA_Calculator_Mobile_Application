import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import time
import json
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class MotionAnalyzer:
    def __init__(self, video_path, height_cm=175):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.height_cm = height_cm  # subject height in cm for calibration
        self.pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.landmarks_history = []
        self.scale_factor = None  # Will be calculated during calibration
       
        # Data for analysis
        self.frame_count = 0
        self.time_points = []
        self.joint_angles = {
            'left_elbow': [], 'right_elbow': [],
            'left_knee': [], 'right_knee': [],
            'left_hip': [], 'right_hip': [],
            'left_ankle': [], 'right_ankle': []
        }
        self.cg_positions = []  # Center of gravity positions
        self.joint_positions = {}
       
    def calibrate(self, frame, landmarks):
        """Calibrate using the subject's height to establish real-world measurements"""
        if landmarks is None or not landmarks.pose_landmarks:
            return None
           
        # Use the distance between head and feet as subject height
        head = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        left_foot = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_foot = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
       
        # Average position of feet
        foot_y = (left_foot.y + right_foot.y) / 2
        pixel_height = (foot_y - head.y) * self.height
       
        # Calculate scale factor (cm per pixel)
        self.scale_factor = self.height_cm / pixel_height
        return self.scale_factor
   
    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
       
        ba = a - b
        bc = c - b
       
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
       
        return np.degrees(angle)
   
    def calculate_center_of_gravity(self, landmarks):
        """Calculate approximate center of gravity"""
        if not landmarks.pose_landmarks:
            return None
           
        # Weighted average of key body points
        # Different body segments have different weights in the COG calculation
        hip_left = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        hip_right = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        shoulder_left = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        shoulder_right = landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
       
        # Simplified COG calculation (can be refined with biomechanical models)
        cog_x = (hip_left.x + hip_right.x + shoulder_left.x + shoulder_right.x) / 4
        cog_y = (hip_left.y + hip_right.y + shoulder_left.y + shoulder_right.y) / 4
       
        return (int(cog_x * self.width), int(cog_y * self.height))
   
    def process_frame(self, frame, draw=True):
        """Process a single frame to detect pose and analyze motion"""
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       
        # Process the frame and get the pose landmarks
        results = self.pose.process(rgb_frame)
       
        # If no calibration yet, perform it
        if self.scale_factor is None and results.pose_landmarks:
            self.calibrate(frame, results)
       
        if results.pose_landmarks:
            # Store landmarks for trajectory analysis
            self.landmarks_history.append(results.pose_landmarks)
           
            # Calculate joint angles
            landmarks = results.pose_landmarks.landmark
           
            # Left elbow angle
            left_elbow = self.calculate_angle(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            )
            self.joint_angles['left_elbow'].append(left_elbow)
           
            # Right elbow angle
            right_elbow = self.calculate_angle(
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            )
            self.joint_angles['right_elbow'].append(right_elbow)
           
            # Left knee angle
            left_knee = self.calculate_angle(
                landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            )
            self.joint_angles['left_knee'].append(left_knee)
           
            # Right knee angle
            right_knee = self.calculate_angle(
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            )
            self.joint_angles['right_knee'].append(right_knee)
           
            # Calculate and store hip angles
            left_hip = self.calculate_angle(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            )
            self.joint_angles['left_hip'].append(left_hip)
           
            right_hip = self.calculate_angle(
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            )
            self.joint_angles['right_hip'].append(right_hip)
           
            # Calculate center of gravity
            cog = self.calculate_center_of_gravity(results)
            if cog:
                self.cg_positions.append(cog)
           
            # Store time point
            self.time_points.append(self.frame_count / self.fps)
           
            # Increment frame counter
            self.frame_count += 1
           
            # Draw pose landmarks and connections
            if draw:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
                )
               
                # Draw center of gravity
                if cog:
                    cv2.circle(frame, cog, 5, (0, 0, 255), -1)
               
                # Draw trajectories
                self.draw_trajectories(frame)
               
                # Display joint angles on frame
                self.display_joint_angles(frame, landmarks)
       
        return frame, results
   
    def draw_trajectories(self, frame):
        """Draw motion trajectories on the frame"""
        # Draw CG trajectory in red
        if len(self.cg_positions) > 1:
            for i in range(1, len(self.cg_positions)):
                cv2.line(frame, self.cg_positions[i-1], self.cg_positions[i], (0, 0, 255), 2)
   
    def display_joint_angles(self, frame, landmarks):
        """Display joint angles on the frame"""
        # Display a few key angles
        if len(self.joint_angles['left_knee']) > 0:
            left_knee_pos = (
                int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * self.width),
                int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * self.height)
            )
            cv2.putText(frame, f"L Knee: {self.joint_angles['left_knee'][-1]:.1f}°",
                       (left_knee_pos[0] + 10, left_knee_pos[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                      
        if len(self.joint_angles['right_knee']) > 0:
            right_knee_pos = (
                int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * self.width),
                int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * self.height)
            )
            cv2.putText(frame, f"R Knee: {self.joint_angles['right_knee'][-1]:.1f}°",
                       (right_knee_pos[0] + 10, right_knee_pos[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
   
    def analyze_video(self, output_path, sample_rate=1):
        """Process the entire video and generate an analyzed version"""
        if not self.cap.isOpened():
            return False
       
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        output_video = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
       
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
               
            # Process every nth frame for efficiency
            if frame_idx % sample_rate == 0:
                processed_frame, _ = self.process_frame(frame)
                output_video.write(processed_frame)
           
            frame_idx += 1
           
        # Release resources
        self.cap.release()
        output_video.release()
       
        return True
   
    def generate_analysis_graphs(self, output_folder):
        """Generate graphs for the motion analysis"""
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
       
        # 1. Joint angle graphs
        plt.figure(figsize=(10, 6))
        for joint, angles in self.joint_angles.items():
            if angles:  # Only plot if we have data
                plt.plot(self.time_points[:len(angles)], angles, label=joint.replace('_', ' ').title())
       
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.title('Joint Angles Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, 'joint_angles.png'))
        plt.close()
       
        # 2. Center of gravity horizontal and vertical position
        if self.cg_positions:
            plt.figure(figsize=(10, 6))
           
            # Extract x and y coordinates
            cg_x = [pos[0] for pos in self.cg_positions]
            cg_y = [pos[1] for pos in self.cg_positions]
           
            # Convert to real-world units if calibrated
            if self.scale_factor:
                cg_x_cm = [x * self.scale_factor for x in cg_x]
                cg_y_cm = [y * self.scale_factor for y in cg_y]
               
                plt.subplot(2, 1, 1)
                plt.plot(self.time_points[:len(cg_x_cm)], cg_x_cm)
                plt.xlabel('Time (s)')
                plt.ylabel('Horizontal Position (cm)')
                plt.title('Center of Gravity - Horizontal Movement')
                plt.grid(True)
               
                plt.subplot(2, 1, 2)
                plt.plot(self.time_points[:len(cg_y_cm)], cg_y_cm)
                plt.xlabel('Time (s)')
                plt.ylabel('Vertical Position (cm)')
                plt.title('Center of Gravity - Vertical Movement')
                plt.grid(True)
            else:
                plt.subplot(2, 1, 1)
                plt.plot(self.time_points[:len(cg_x)], cg_x)
                plt.xlabel('Time (s)')
                plt.ylabel('Horizontal Position (px)')
                plt.title('Center of Gravity - Horizontal Movement')
                plt.grid(True)
               
                plt.subplot(2, 1, 2)
                plt.plot(self.time_points[:len(cg_y)], cg_y)
                plt.xlabel('Time (s)')
                plt.ylabel('Vertical Position (px)')
                plt.title('Center of Gravity - Vertical Movement')
                plt.grid(True)
           
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'cg_position.png'))
            plt.close()
           
            # 3. CG trajectory plot (bird's eye view)
            plt.figure(figsize=(8, 8))
            plt.plot(cg_x, cg_y)
            plt.scatter(cg_x[0], cg_y[0], color='green', s=100, label='Start')
            plt.scatter(cg_x[-1], cg_y[-1], color='red', s=100, label='End')
            plt.xlabel('Horizontal Position')
            plt.ylabel('Vertical Position')
            plt.title('Center of Gravity Trajectory')
            plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_folder, 'cg_trajectory.png'))
            plt.close()
       
        return True
   
    def export_data(self, output_path):
        """Export the analysis data to CSV"""
        # Prepare data for export
        data = {
            'Time (s)': self.time_points
        }
       
        # Add joint angles
        for joint, angles in self.joint_angles.items():
            data[f'{joint.replace("_", " ").title()} Angle'] = angles + [None] * (len(self.time_points) - len(angles))
       
        # Add CG positions if available
        if self.cg_positions:
            cg_x = [pos[0] for pos in self.cg_positions]
            cg_y = [pos[1] for pos in self.cg_positions]
           
            # Convert to real-world units if calibrated
            if self.scale_factor:
                data['CG X (cm)'] = [x * self.scale_factor for x in cg_x] + [None] * (len(self.time_points) - len(cg_x))
                data['CG Y (cm)'] = [y * self.scale_factor for y in cg_y] + [None] * (len(self.time_points) - len(cg_y))
            else:
                data['CG X (px)'] = cg_x + [None] * (len(self.time_points) - len(cg_x))
                data['CG Y (px)'] = cg_y + [None] * (len(self.time_points) - len(cg_y))
       
        # Create DataFrame and export
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
       
        return True

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
   
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
   
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
       
        # Get height parameter for calibration
        height_cm = float(request.form.get('height', 175))
       
        # Generate unique ID for this analysis
        analysis_id = str(int(time.time()))
       
        # Create results directory
        result_dir = os.path.join(RESULTS_FOLDER, analysis_id)
        os.makedirs(result_dir, exist_ok=True)
       
        # Create output paths
        output_video_path = os.path.join(result_dir, 'analyzed_' + filename)
        output_data_path = os.path.join(result_dir, 'data.csv')
       
        # Process the video
        analyzer = MotionAnalyzer(filepath, height_cm)
        analyzer.analyze_video(output_video_path)
        analyzer.generate_analysis_graphs(result_dir)
        analyzer.export_data(output_data_path)
       
        # Create a results object with paths
        results = {
            'analysis_id': analysis_id,
            'video_path': '/results/' + analysis_id + '/analyzed_' + filename,
            'data_path': '/results/' + analysis_id + '/data.csv',
            'joint_angles_graph': '/results/' + analysis_id + '/joint_angles.png',
            'cg_position_graph': '/results/' + analysis_id + '/cg_position.png',
            'cg_trajectory_graph': '/results/' + analysis_id + '/cg_trajectory.png'
        }
       
        # Save results metadata
        with open(os.path.join(result_dir, 'metadata.json'), 'w') as f:
            json.dump(results, f)
       
        return jsonify(results), 200
   
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/results/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    # Load metadata
    metadata_path = os.path.join(RESULTS_FOLDER, analysis_id, 'metadata.json')
   
    if not os.path.exists(metadata_path):
        return jsonify({'error': 'Analysis not found'}), 404
   
    with open(metadata_path, 'r') as f:
        results = json.load(f)
   
    return jsonify(results), 200

@app.route('/results/<analysis_id>/<path:resource>', methods=['GET'])
def get_result_file(analysis_id, resource):
    file_path = os.path.join(RESULTS_FOLDER, analysis_id, resource)
   
    if not os.path.exists(file_path):
        return jsonify({'error': 'Resource not found'}), 404
   
    return send_file(file_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

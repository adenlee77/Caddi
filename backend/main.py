from flask import Flask, request, jsonify, session, render_template
from flask_cors import CORS
import tempfile
import cv2
import mediapipe as mp
import numpy as np
from pose_utils import *
from swing_stage import SwingPhaseDetector
from feedback import swing_feedback

app = Flask(__name__)
CORS(app)
mp_pose = mp.solutions.pose

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    video = request.files['video']
    club = request.form.get('club', 'unknown club')

    # Save to a temporary file-like object (auto-deleted)
    with tempfile.NamedTemporaryFile(suffix=".webm") as tmp:
        tmp.write(video.read())
        tmp.flush()

        cap = cv2.VideoCapture(tmp.name)
        swing_detector = SwingPhaseDetector()
        swing_sequence = []

        with mp_pose.Pose() as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                if not results.pose_landmarks:
                    continue

                try:
                    landmarks = results.pose_landmarks.landmark
                    ball_pos = get_ball_pos(frame)
                    left_knee_angle, left_knee_point = get_left_knee_angle(landmarks)
                    left_elbow_angle, left_elbow_point = get_left_elbow_angle(landmarks)
                    right_knee_angle, right_knee_point = get_right_knee_angle(landmarks)
                    right_elbow_angle, right_elbow_point = get_right_elbow_angle(landmarks)
                    spine_angle, shoulder_center = get_spine_angle(landmarks)

                    h, w, _ = frame.shape
                    swing_state = swing_detector.update(landmarks)

                    if swing_state != 'waiting':
                        swing_sequence.append({
                            "angles": {
                                "left_knee_angle": int(left_knee_angle),
                                "right_knee_angle": int(right_knee_angle),
                                "left_elbow_angle": int(left_elbow_angle),
                                "right_elbow_angle": int(right_elbow_angle),
                                "spine_angle": int(spine_angle),
                            },
                            "positions": {
                                "left_knee": tuple(np.multiply(left_knee_point, [w, h]).astype(int)),
                                "right_knee": tuple(np.multiply(right_knee_point, [w, h]).astype(int)),
                                "left_elbow": tuple(np.multiply(left_elbow_point, [w, h]).astype(int)),
                                "right_elbow": tuple(np.multiply(right_elbow_point, [w, h]).astype(int)),
                                "shoulder_center": tuple(np.multiply(shoulder_center, [w, h]).astype(int)),
                            },
                            "ball_position": ball_pos,
                            "frame_index": len(swing_sequence),
                            "phase": swing_state
                        })

                    if swing_state == 'End':
                        break
                except Exception as e:
                    print("Error processing frame:", e)
                    continue

        cap.release()

    # Prepare prompt and feedback
    swing_data = {
        "club": club,
        "view": "front-facing",
        "swing_sequence": swing_sequence
    }

    prompt = f"""
                You are a golf coach. Here's a player's front-facing swing data captured frame-by-frame using a {club}:
                {swing_data}

                Please analyze the swing and respond breifly in **JSON format** with the following fields:
                - "overall_feedback": A concise summary highlighting key strengths or weaknesses observed in the swing. Focus on the most notable aspects, whether positive or negative. Please mention club specific issues or strengths. Make the response roughly 2-3 sentences.
                - "swing_grade": A numerical score (0–100) reflecting the overall quality of the swing using a standardized method. Use the full range of 0-100 and make the scores **EXTREMELY VARIABLE**. This goes both ways, good and bad based on small mistakes or small good actions, please make sure even **minor differences in swing data** should cause **significant changes** in the score. Use the full 0–100 scale, but do not grade harshly unless there are **major visible faults** or the data is incomplete, an average swing should have a score of around 80.

                
                
                - "drills": A list of 1–2 targeted drills that address the most significant areas for improvement. Each drill must be formatted as an object with two fields: "title" (a short, descriptive name) and "description" (a brief explanation of how to set up and perform the drill). Ensure the drills focus on overall swing improvement, not just isolated mechanics like wrist movement or hip rotation. Avoid repetition and make sure each drill is meaningfully distinct from the other

                Respond with only a valid JSON object. Do not wrap it in code block formatting, no explanation or extra commentary.
                """

    feedback = swing_feedback(prompt)
    return jsonify(feedback)

@app.route('/feedback')
def results():
    return render_template('feedback.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
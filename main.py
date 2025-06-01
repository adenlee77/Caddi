import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
from pose_utils import (
    get_left_knee_angle,
    get_right_elbow_angle,
    get_left_elbow_angle,
    get_right_knee_angle,
    get_spine_angle
)
from swing_stage import SwingPhaseDetector
from feedback import swing_feedback

cap = cv2.VideoCapture(1)
swing_detector = SwingPhaseDetector()
feedback = swing_feedback()

# Set up mediapipe instance
with mp_pose.Pose() as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Change colour to RGB, make a detection in RGB, then switch back to BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Get exact landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            left_knee_angle, left_knee_point = get_left_knee_angle(landmarks)
            left_elbow_angle, left_elbow_point = get_left_elbow_angle(landmarks)
            right_knee_angle, right_knee_point = get_right_knee_angle(landmarks)
            right_elbow_angle, right_elbow_point = get_right_elbow_angle(landmarks)
            spine_angle, shoulder_center = get_spine_angle(landmarks)

            # Get dimensions from the frame
            frame_height, frame_width, _ = image.shape
            
            # Display angle on the joint
            cv2.putText(image, f"{int(left_knee_angle)}",
                        tuple(np.multiply(left_knee_point, [frame_width, frame_height]).astype(int)),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (250, 60, 5), 2)
            
            cv2.putText(image, f"{int(right_knee_angle)}",
                        tuple(np.multiply(right_knee_point, [frame_width, frame_height]).astype(int)),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (250, 60, 5), 2)

            cv2.putText(image, f"{int(left_elbow_angle)}",
                        tuple(np.multiply(left_elbow_point, [frame_width, frame_height]).astype(int)),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (250, 60, 5), 2)
            
            cv2.putText(image, f"{int(right_elbow_angle)}",
                        tuple(np.multiply(right_elbow_point, [frame_width, frame_height]).astype(int)),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (250, 60, 5), 2)

            cv2.putText(image, f"{int(spine_angle)}",
                tuple(np.multiply(shoulder_center, [frame_width, frame_height]).astype(int)),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (250, 60, 5), 2)
            
            swing_state = swing_detector.update(landmarks)
            cv2.putText(image, swing_state, (30, 50),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 255), 2)
            
            print(swing_state)

            keypoints = {
                "left_knee": tuple(np.multiply(left_knee_point, [frame_width, frame_height]).astype(int)),
                "right_knee": tuple(np.multiply(right_knee_point, [frame_width, frame_height]).astype(int)),
                "left_elbow": tuple(np.multiply(left_elbow_point, [frame_width, frame_height]).astype(int)),
                "right_elbow": tuple(np.multiply(right_elbow_point, [frame_width, frame_height]).astype(int)),
                "shoulder_center": tuple(np.multiply(shoulder_center, [frame_width, frame_height]).astype(int)),
            }

            keyangles = {
                "left_knee_angle": int(left_knee_angle),
                "right_knee_angle": int(right_knee_angle),
                "left_elbow_angle": int(left_elbow_angle),
                "right_elbow_angle": int(right_elbow_angle),
                "spine_angle": int(spine_angle),
            }

            swing_feedback.add(swing_state, keypoints, keyangles)

            if swing_state == 'End':
                break

        except Exception as e:
            print(f"Failed due to error: {e}")
            import traceback
            traceback.print_exc()
            pass

        # Draw detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Webcam Feed', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows


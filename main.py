import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
from angle import calculate_angle

cap = cv2.VideoCapture(1)

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
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEF_ANKLE.value].y]

            # Get dimensions from the frame
            frame_height, frame_width, _ = image.shape

            angle = calculate_angle(hip, knee, ankle)

            # Display angle on the joint
            cv2.putText(image, str(angle), tuple(np.multiply(knee, [frame_width, frame_height]).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 4, cv2.LINE_AA)
        except:
            pass

        # Draw detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Webcam Feed', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows


from collections import deque
import numpy as np
import mediapipe as mp

POSE = mp.solutions.pose

class SwingPhaseDetector:
    def __init__(self,
                 history_length=10,
                 stability_std_threshold=0.003,
                 wrist_movement_threshold=0.02,
                 shoulder_z_threshold=0.1):
        self.l_wrist_y_history = deque(maxlen=history_length)
        self.r_wrist_y_history = deque(maxlen=history_length)

        self.setup_baseline_left_y = None
        self.setup_baseline_right_y = None

        self.shoulder_z_baseline = None

        self.mode = "waiting"
        self.mp_pose = mp.solutions.pose

        self.stability_std_threshold = stability_std_threshold
        self.wrist_movement_threshold = wrist_movement_threshold
        self.shoulder_z_threshold = shoulder_z_threshold

    def update(self, landmarks):

        # Grab landmarks
        l_wrist_y = landmarks[POSE.PoseLandmark.LEFT_WRIST.value].y
        r_wrist_y = landmarks[POSE.PoseLandmark.RIGHT_WRIST.value].y
        l_shoulder_z = landmarks[POSE.PoseLandmark.LEFT_SHOULDER.value].z
        r_shoulder_z = landmarks[POSE.PoseLandmark.RIGHT_SHOULDER.value].z

        # Add to the deque of history
        self.l_wrist_y_history.append(l_wrist_y)
        self.r_wrist_y_history.append(r_wrist_y)

        if len(self.l_wrist_y_history) < self.l_wrist_y_history.maxlen:
            return self.mode

        # Find standard deviation to see if wrists are still
        l_std = np.std(self.l_wrist_y_history)
        r_std = np.std(self.r_wrist_y_history)

        if self.mode == "waiting":
            if l_std < self.stability_std_threshold and r_std < self.stability_std_threshold:

                # Set points for where wrists rest and shoulders depth difference
                self.setup_baseline_left_y = np.mean(self.l_wrist_y_history)
                self.setup_baseline_right_y = np.mean(self.r_wrist_y_history)
                self.shoulder_z_baseline = abs(l_shoulder_z - r_shoulder_z)
                self.mode = "setup_detected"

        elif self.mode == "setup_detected":

            # Check if wrists move from rest position and shoulders rotate
            left_wrist_moved = abs(self.setup_baseline_left_y - l_wrist_y) > self.wrist_movement_threshold
            right_wrist_moved = abs(self.setup_baseline_right_y - r_wrist_y) > self.wrist_movement_threshold
            shoulder_rotated = abs(l_shoulder_z - r_shoulder_z) - self.shoulder_z_baseline > self.shoulder_z_threshold

            if left_wrist_moved and right_wrist_moved and shoulder_rotated:
                self.mode = "backswing_started"

        return self.mode

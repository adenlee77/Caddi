from collections import deque
import numpy as np
import mediapipe as mp

POSE = mp.solutions.pose

class SwingPhaseDetector:
    def __init__(self,
                 history_length=5,
                 stability_std_threshold=0.003,
                 wrist_movement_threshold=0.02,
                 shoulder_z_threshold=0.05,
                 downswing_wrist_threshold=0.005,
                 original_spot_threshold=0.1):
        self.l_wrist_y_history = deque(maxlen=history_length)
        self.r_wrist_y_history = deque(maxlen=history_length)

        self.setup_baseline_left_y = None
        self.setup_baseline_right_y = None
        self.shoulder_z_baseline = None

        self.back_peak_left_y = None
        self.back_peak_right_y = None
        self.prev_left_y = None
        self.prev_right_y = None
        self.peak_left_y = float('-inf')
        self.peak_right_y = float('-inf')

        self.mode = "waiting"
        self.mp_pose = mp.solutions.pose

        self.stability_std_threshold = stability_std_threshold
        self.wrist_movement_threshold = wrist_movement_threshold
        self.shoulder_z_threshold = shoulder_z_threshold
        self.downswing_wrist_threshold = downswing_wrist_threshold
        self.original_spot_threshold = original_spot_threshold

    def update(self, landmarks):

        # Grab landmarks
        l_wrist_y = landmarks[POSE.PoseLandmark.LEFT_WRIST.value].y
        r_wrist_y = landmarks[POSE.PoseLandmark.RIGHT_WRIST.value].y
        l_shoulder_z = landmarks[POSE.PoseLandmark.LEFT_SHOULDER.value].z
        r_shoulder_z = landmarks[POSE.PoseLandmark.RIGHT_SHOULDER.value].z
        shoulder_z_diff = abs(l_shoulder_z - r_shoulder_z)

        # Add to the deque of history
        self.l_wrist_y_history.append(l_wrist_y)
        self.r_wrist_y_history.append(r_wrist_y)

        if len(self.l_wrist_y_history) < self.l_wrist_y_history.maxlen:
            return self.mode

        # Find standard deviation to see if wrists are still
        l_std = np.std(self.l_wrist_y_history)
        r_std = np.std(self.r_wrist_y_history)

        # Change from waiting to setup
        if self.mode == "waiting":
            if l_std < self.stability_std_threshold and r_std < self.stability_std_threshold:

                # Set points for where wrists rest and shoulders depth difference
                self.setup_baseline_left_y = np.mean(self.l_wrist_y_history)
                self.setup_baseline_right_y = np.mean(self.r_wrist_y_history)
                self.shoulder_z_baseline = abs(l_shoulder_z - r_shoulder_z)
                self.mode = "setup"

        # Change from setup to backswing
        elif self.mode == "setup":

            # Check if wrists move from rest position and shoulders rotate
            left_wrist_moved = abs(self.setup_baseline_left_y - l_wrist_y) > self.wrist_movement_threshold
            right_wrist_moved = abs(self.setup_baseline_right_y - r_wrist_y) > self.wrist_movement_threshold
            shoulder_rotated = abs(l_shoulder_z - r_shoulder_z) - self.shoulder_z_baseline > self.shoulder_z_threshold

            if left_wrist_moved and right_wrist_moved and shoulder_rotated:
                self.back_peak_left_y = l_wrist_y
                self.back_peak_right_y = r_wrist_y
                self.mode = "backswing"

        # Change from backswing to downswing
        elif self.mode == "backswing":

            # Check peak y-value of backswing
            self.peak_left_y = min(self.peak_left_y, l_wrist_y)
            self.peak_right_y = min(self.peak_right_y, r_wrist_y)

            # Calculate average slope of wrist movement
            left_slope = np.mean(np.diff(list(self.l_wrist_y_history)))
            right_slope = np.mean(np.diff(list(self.r_wrist_y_history)))

            # Is wrist falling (positive slope)?
            left_dropping = left_slope > 0.002
            right_dropping = right_slope > 0.002

            # Has wrist dropped significantly from its peak?
            left_drop_passed = l_wrist_y > self.peak_left_y + self.downswing_wrist_threshold
            right_drop_passed = r_wrist_y > self.peak_right_y + self.downswing_wrist_threshold

            # Is shoulder rotating back?
            shoulder_reversing = abs(l_shoulder_z - r_shoulder_z) < self.shoulder_z_baseline + self.shoulder_z_threshold

            if left_dropping and right_dropping and left_drop_passed and right_drop_passed and shoulder_reversing:
                self.mode = "downswing"
        
        # Change from downswing to followthrough
        elif self.mode == "downswing":

            # See if wrists go back to original setup position
            left_back = abs(l_wrist_y - self.setup_baseline_left_y) < self.original_spot_threshold
            right_back = abs(r_wrist_y - self.setup_baseline_right_y) < self.original_spot_threshold

            # Check if left wrist is going up
            left_rising = (
                self.prev_left_y is not None and
                l_wrist_y < self.prev_left_y and
                l_wrist_y < self.back_peak_left_y + self.downswing_wrist_threshold
            )

            # Check if right wrist is going up
            right_rising = (
                self.prev_right_y is not None and
                r_wrist_y < self.prev_right_y and
                r_wrist_y < self.back_peak_right_y + self.downswing_wrist_threshold
            )

            if (left_back and right_back) or (left_rising and right_rising):
                self.mode = "followthrough"
        
        # Change from followthrough to end of swing
        elif self.mode == "followthrough":
            
            # Check if left wrist is going down
            left_dropping = (
                self.prev_left_y is not None and
                l_wrist_y > self.prev_left_y and
                l_wrist_y > self.back_peak_left_y + self.downswing_wrist_threshold
            )

            # Check if right wrist is going down
            right_dropping = (
                self.prev_right_y is not None and
                r_wrist_y > self.prev_right_y and
                r_wrist_y > self.back_peak_right_y + self.downswing_wrist_threshold
            )

            # Check for wrists staying still
            if (l_std < self.stability_std_threshold and r_std < self.stability_std_threshold) or (left_dropping and right_dropping):
                self.mode = "End"

        self.prev_left_y = l_wrist_y
        self.prev_right_y = r_wrist_y

        return self.mode

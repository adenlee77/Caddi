import numpy as np
import mediapipe as mp

POSE = mp.solutions.pose

# Calculates the angle between three landmarks
def calculate_angle(first, mid, last):
    first = np.array(first)
    mid = np.array(mid)
    last = np.array(last)

    radians = np.arctan2(last[1] - mid[1], last[0] - mid[0]) - np.arctan2(first[1] - mid[1], first[0] - mid[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180:
        angle = 360 - angle
    
    return angle

def get_left_knee_angle(landmarks):
    hip = [landmarks[POSE.PoseLandmark.LEFT_HIP.value].x,
           landmarks[POSE.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[POSE.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[POSE.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[POSE.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[POSE.PoseLandmark.LEFT_ANKLE.value].y]
    return calculate_angle(hip, knee, ankle), knee


def get_right_knee_angle(landmarks):
    hip = [landmarks[POSE.PoseLandmark.RIGHT_HIP.value].x,
           landmarks[POSE.PoseLandmark.RIGHT_HIP.value].y]
    knee = [landmarks[POSE.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[POSE.PoseLandmark.RIGHT_KNEE.value].y]
    ankle = [landmarks[POSE.PoseLandmark.RIGHT_ANKLE.value].x,
             landmarks[POSE.PoseLandmark.RIGHT_ANKLE.value].y]
    return calculate_angle(hip, knee, ankle), knee


def get_left_elbow_angle(landmarks):
    shoulder = [landmarks[POSE.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[POSE.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[POSE.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[POSE.PoseLandmark.LEFT_ELBOW.value].y]
    wrist = [landmarks[POSE.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[POSE.PoseLandmark.LEFT_WRIST.value].y]
    return calculate_angle(shoulder, elbow, wrist), elbow

def get_right_elbow_angle(landmarks):
    shoulder = [landmarks[POSE.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[POSE.PoseLandmark.RIGHT_SHOULDER.value].y]
    elbow = [landmarks[POSE.PoseLandmark.RIGHT_ELBOW.value].x,
             landmarks[POSE.PoseLandmark.RIGHT_ELBOW.value].y]
    wrist = [landmarks[POSE.PoseLandmark.RIGHT_WRIST.value].x,
             landmarks[POSE.PoseLandmark.RIGHT_WRIST.value].y]
    return calculate_angle(shoulder, elbow, wrist), elbow


def get_spine_angle(landmarks):
    l_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                  landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
    r_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                  landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y]

    shoulder_vector = np.array(r_shoulder) - np.array(l_shoulder)

    dx = shoulder_vector[0]
    dy = shoulder_vector[1]

    angle = abs(np.degrees(np.arctan2(dy, dx)))
    return angle, [(l_shoulder[0] + r_shoulder[0]) / 2,
                   (l_shoulder[1] + r_shoulder[1]) / 2]
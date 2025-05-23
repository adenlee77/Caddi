import numpy as np

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
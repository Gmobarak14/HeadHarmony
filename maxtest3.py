import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from pythonosc import udp_client
import time

# === OSC Setup ===
client = udp_client.SimpleUDPClient("127.0.0.1", 7400)

# === MediaPipe Setup ===
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# === Webcam ===
cap = cv2.VideoCapture(0)

# --- Buffers for smoothing ---
yaw_buffer = deque(maxlen=5)
pitch_buffer = deque(maxlen=5)

# --- Compute head rotation (2D version) ---
def get_head_rotation(landmarks):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose_tip = landmarks[1]

    mid_eye_x = (left_eye.x + right_eye.x) / 2
    mid_eye_y = (left_eye.y + right_eye.y) / 2

    # 2D yaw and pitch (scaled)
    yaw = (nose_tip.x - mid_eye_x) * 100
    pitch = (mid_eye_y - nose_tip.y) * 100

    return yaw, pitch

# --- Trigger thresholds ---
THRESHOLD_YAW = 5
THRESHOLD_PITCH = 5
NEUTRAL_HYST = 2

# --- Display settings ---
DISPLAY_TIME = 1.0  # seconds to show on-screen direction

# --- Trigger state ---
trigger_state = {"up": True, "down": True, "left": True, "right": True}
last_trigger_time = {"up": 0, "down": 0, "left": 0, "right": 0}

# --- Calibration ---
print("Keep your head straight for calibration (2 seconds)...")
baseline_yaw_list = []
baseline_pitch_list = []
calibration_start = time.time()
CALIBRATION_TIME = 2.0

while time.time() - calibration_start < CALIBRATION_TIME:
    ret, frame = cap.read()
    if not ret:
        continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        yaw, pitch = get_head_rotation(face.landmark)
        baseline_yaw_list.append(yaw)
        baseline_pitch_list.append(pitch)

baseline_yaw = float(np.mean(baseline_yaw_list))
baseline_pitch = float(np.mean(baseline_pitch_list))
print(f"Calibration done. Neutral Yaw: {baseline_yaw:.2f}, Pitch: {baseline_pitch:.2f}")
print("Move your head to trigger directions.")

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    current_time = time.time()

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        yaw, pitch = get_head_rotation(face.landmark)

        # --- Smooth ---
        yaw_buffer.append(yaw)
        pitch_buffer.append(pitch)
        yaw_s = float(np.mean(yaw_buffer))
        pitch_s = float(np.mean(pitch_buffer))

        # --- Deviation from baseline ---
        yaw_dev = yaw_s - baseline_yaw
        pitch_dev = pitch_s - baseline_pitch

        # --- UP ---
        if pitch_dev >= THRESHOLD_PITCH and trigger_state["up"]:
            client.send_message("/head/up", pitch_dev)
            trigger_state["up"] = False
            last_trigger_time["up"] = current_time
        elif pitch_dev < THRESHOLD_PITCH - NEUTRAL_HYST:
            trigger_state["up"] = True

        # --- DOWN ---
        if pitch_dev <= -THRESHOLD_PITCH and trigger_state["down"]:
            client.send_message("/head/down", pitch_dev)
            trigger_state["down"] = False
            last_trigger_time["down"] = current_time
        elif pitch_dev > -THRESHOLD_PITCH + NEUTRAL_HYST:
            trigger_state["down"] = True

        # --- RIGHT ---
        if yaw_dev >= THRESHOLD_YAW and trigger_state["right"]:
            client.send_message("/head/right", yaw_dev)
            trigger_state["right"] = False
            last_trigger_time["right"] = current_time
        elif yaw_dev < THRESHOLD_YAW - NEUTRAL_HYST:
            trigger_state["right"] = True

        # --- LEFT ---
        if yaw_dev <= -THRESHOLD_YAW and trigger_state["left"]:
            client.send_message("/head/left", yaw_dev)
            trigger_state["left"] = False
            last_trigger_time["left"] = current_time
        elif yaw_dev > -THRESHOLD_YAW + NEUTRAL_HYST:
            trigger_state["left"] = True

        # --- On-screen labels ---
        cv2.putText(frame, f"Yaw dev: {yaw_dev:.1f}  Pitch dev: {pitch_dev:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Color-coded display
        y_offset = 70
        direction_colors = {
            "up": (0, 255, 0),     # Green
            "down": (0, 0, 255),   # Red
            "left": (255, 0, 0),   # Blue
            "right": (0, 255, 255) # Yellow
        }
        for dir_name, color in direction_colors.items():
            if current_time - last_trigger_time[dir_name] <= DISPLAY_TIME:
                cv2.putText(frame, f"{dir_name.upper()}",
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                y_offset += 50  # stack vertically

    cv2.imshow("Head Tracking (ESC to quit)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

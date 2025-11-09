import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

pyautogui.FAILSAFE = False

# --- 1. CORE PARAMETER CHANGES ---

# SMOOTHING:
SMOOTHING_ALPHA = 0.2
smoothed_yaw = None
smoothed_pitch = None

# MAPPING:
YAW_RANGE = [-45, 45]  # [Min angle Left, Max angle Right]
PITCH_RANGE = [-35, 25]  # [Min angle Down, Max angle Up]

# --- NEW CLICK LOGIC ---
# We now track the time of the last click to detect a double-click
fist_state = "OPEN"  # Possible states: "OPEN", "CLOSED"
last_click_time = 0
DOUBLE_CLICK_INTERVAL = 0.5  # Time in seconds for a double click

# SCROLL LOGIC
scroll_prev_y = None

# --- 2. MEDIAPIPE & CAMERA SETUP ---

print("Loading MediaPipe models...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 3D model points
MEDIAPIPE_LANDMARK_IDS = [1, 133, 362, 234, 454, 152]
MEDIAPIPE_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),    # Nose Tip
    (-30.0, -30.0, -30.0),  # Left Eye
    (30.0, -30.0, -30.0),   # Right Eye
    (-60.0, 60.0, -60.0),   # Left Ear
    (60.0, 60.0, -60.0),    # Right Ear
    (0.0, 100.0, -100.0)  # Chin
], dtype=np.float32)

# Camera parameters
FOCAL_LENGTH = 1
CENTER = (0, 0)

# --- 3. HELPER FUNCTIONS ---

def rotation_matrix_to_euler_angles(R):
    """ Converts a rotation matrix to Euler angles (yaw, pitch, roll) """
    yaw = np.arctan2(R[1, 0], R[0, 0]) * (180.0 / np.pi)
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)) * (180.0 / np.pi)
    roll = np.arctan2(R[2, 1], R[2, 2]) * (180.0 / np.pi)
    return yaw, pitch, roll

def estimate_head_pose(img_rgb):
    """ Estimates head pose using MediaPipe FaceMesh """
    global FOCAL_LENGTH, CENTER
    ih, iw = img_rgb.shape[:2]
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]
    
    image_points = np.array([
        [face_landmarks.landmark[i].x * iw, face_landmarks.landmark[i].y * ih]
        for i in MEDIAPIPE_LANDMARK_IDS
    ], dtype=np.float32)

    FOCAL_LENGTH = iw
    CENTER = (iw / 2, ih / 2)
    CAMERA_MATRIX = np.array([[FOCAL_LENGTH, 0, CENTER[0]], [0, FOCAL_LENGTH, CENTER[1]], [0, 0, 1]], dtype=np.float32)
    DIST_COEFFS = np.zeros((4, 1))

    success, rotation_vector, _ = cv2.solvePnP(MEDIAPIPE_MODEL_POINTS, image_points, CAMERA_MATRIX, DIST_COEFFS)

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix)
    
    return {"yaw": -yaw, "pitch": pitch, "roll": roll}

def draw_info(img, yaw, pitch, fps, status):
    """ Draws all info text on the frame """
    cv2.putText(img, f"Yaw: {yaw:.2f}", (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(img, f"Pitch: {pitch:.2f}", (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(img, f"FPS: {fps:.2f}", (20, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(img, f"ACTION: {status}", (20, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

# --- 4. MAIN WEBCAM LOOP ---

print("Starting Controller...")
print("PERFORMANCE FIX: Setting camera to 640x480")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

screen_width, screen_height = pyautogui.size()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    start_time = time.time()
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    status_text = ""
    gesture_detected = False

    # --- HAND GESTURE DETECTION (Hands Model) ---
    hand_results = hands.process(img_rgb)
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            
            # Fist Detection Logic
            try:
                is_fist = (
                    landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
                    landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                    landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y and
                    landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_PIP].y
                )
            except:
                is_fist = False

            # Pinch Detection Logic
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pinch_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
            is_pinch = pinch_distance < 0.05
            
            # --- UPDATED CLICK & SCROLL LOGIC ---
            if is_fist:
                gesture_detected = True
                current_time = time.time()

                if fist_state == "OPEN":
                    # This is the EVENT of closing the fist
                    
                    if (current_time - last_click_time) < DOUBLE_CLICK_INTERVAL:
                        # --- THIS IS A DOUBLE CLICK ---
                        status_text = "DOUBLE CLICK!"
                        print("DOUBLE CLICK!")
                        pyautogui.doubleClick()
                        # Reset time so the next click isn't a double
                        last_click_time = 0 
                    else:
                        # --- THIS IS A SINGLE CLICK ---
                        status_text = "CLICK!"
                        print("CLICK!")
                        pyautogui.click()
                        # Record the time of this single click
                        last_click_time = current_time

                    fist_state = "CLOSED" # Update state
                
                else:
                    # The fist is just being held
                    status_text = "Fist (Held)"
            
            elif is_pinch:
                gesture_detected = True
                index_tip_y = index_tip.y
                
                if scroll_prev_y is not None:
                    scroll_delta = index_tip_y - scroll_prev_y
                    if abs(scroll_delta) > 0.01:
                        if scroll_delta > 0:
                            status_text = "Scrolling DOWN"
                            pyautogui.scroll(-20)
                        else:
                            status_text = "Scrolling UP"
                            pyautogui.scroll(20)
                
                scroll_prev_y = index_tip_y
            
            else:
                # No gesture detected, reset states
                fist_state = "OPEN"
                scroll_prev_y = None
    else:
        # No hand, reset states
        fist_state = "OPEN"
        scroll_prev_y = None

    # --- HEAD POSE DETECTION (Face Mesh Model) ---
    gaze_data = estimate_head_pose(img_rgb)
    
    current_yaw, current_pitch = 0, 0
    
    if gaze_data:
        if smoothed_yaw is None:
            smoothed_yaw = gaze_data["yaw"]
            smoothed_pitch = gaze_data["pitch"]
        else:
            smoothed_yaw = (SMOOTHING_ALPHA * gaze_data["yaw"]) + ((1 - SMOOTHING_ALPHA) * smoothed_yaw)
            smoothed_pitch = (SMOOTHING_ALPHA * gaze_data["pitch"]) + ((1 - SMOOTHING_ALPHA) * smoothed_pitch)

        current_yaw = smoothed_yaw
        current_pitch = smoothed_pitch

        # --- MOUSE MOVEMENT ---
        if not gesture_detected:
            if status_text == "":
                status_text = "Moving"

            target_x = np.interp(smoothed_yaw, YAW_RANGE, [screen_width - 1, 0])
            target_y = np.interp(smoothed_pitch, PITCH_RANGE, [screen_height - 1, 0])
            
            target_x = np.clip(target_x, 0, screen_width - 1)
            target_y = np.clip(target_y, 0, screen_height - 1)
                
            pyautogui.moveTo(target_x, target_y)

    # --- DISPLAY ---
    processing_time = time.time() - start_time
    fps = 1 / processing_time if processing_time > 0 else float('inf')
    
    draw_info(frame, current_yaw, current_pitch, fps, status_text)
    cv2.imshow("Head Pose Mouse Controller", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
hands.close()

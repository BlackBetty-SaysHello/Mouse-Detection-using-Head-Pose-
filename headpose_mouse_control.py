import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time


pyautogui.FAILSAFE = False



# MediaPipe Face Mesh (Head-Pose-to-Move) 
print("Loading MediaPipe Face Mesh model")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
MEDIAPIPE_LANDMARK_IDS = [1, 133, 362, 234, 454, 152]
MEDIAPIPE_MODEL_POINTS = np.array([
    ( 0.0,    0.0,     0.0),    # Nose Tip
    (-30.0, -30.0,   -30.0),    # Left Eye
    ( 30.0, -30.0,   -30.0),    # Right Eye
    (-60.0,  60.0,   -60.0),    # Left Ear
    ( 60.0,  60.0,   -60.0),    # Right Ear
    ( 0.0,  100.0,  -100.0)    # Chin
], dtype=np.float32)

# MediaPipe Hands (Gestures)
print("Loading MediaPipe Hands model")
mp_hands = mp.solutions.hands
#  Increased confidence to prevent false gestures
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# SHARED PARAMETERS & FUNCTIONS 

# Smoothing parameters
alpha = 0.7 
smoothed_yaw = None
smoothed_pitch = None

# Gesture logic
prev_y = None 
last_click_time = 0
CLICK_COOLDOWN = 1.0 

# Camera Parameters
FOCAL_LENGTH = 1
CENTER = (0, 0)

def rotation_matrix_to_euler_angles(R):
    """ Converts a rotation matrix to Euler angles (yaw, pitch, roll) """
    yaw = np.arctan2(R[1, 0], R[0, 0]) * (180.0 / np.pi)
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)) * (180.0 / np.pi) 
    roll = np.arctan2(R[2, 1], R[2, 2]) * (180.0 / np.pi)
    return yaw, pitch, roll

def draw_info(img, gaze, fps, status):
    """ Draws the gaze arrow and all info text on the frame """
    if gaze: 
        face = gaze["face"]
        # Draw Gaze Arrow
        arrow_length = img.shape[1] / 3
        dx = -arrow_length * np.sin(np.radians(gaze["yaw"]))
        dy = -arrow_length * np.sin(np.radians(gaze["pitch"]))
        cv2.arrowedLine(img, (int(face["x"]), int(face["y"])), (int(face["x"] + dx), int(face["y"] + dy)), (0, 0, 255), 2, cv2.LINE_AA)
        
        
        cv2.putText(img, f"Yaw: {gaze['yaw']:.2f}", (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(img, f"Pitch: {gaze['pitch']:.2f}", (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(img, f"Roll: {gaze['roll']:.2f}", (20, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    
    # Framework & FPS
    cv2.putText(img, "Framework: MediaPipe", (20, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(img, f"FPS: {fps:.2f}", (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Status Text
    if status:
        cv2.putText(img, f"ACTION: {status}", (20, 180), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

def estimate_head_pose_mediapipe(img_rgb):
    """ Estimates head pose using MediaPipe """
    global FOCAL_LENGTH, CENTER
    results = face_mesh.process(img_rgb)
    
    if not results.multi_face_landmarks:
        return None
        
    face_landmarks = results.multi_face_landmarks[0]
    ih, iw = img_rgb.shape[:2]
    
    image_points = np.array([
        [face_landmarks.landmark[i].x * iw, face_landmarks.landmark[i].y * ih]
        for i in MEDIAPIPE_LANDMARK_IDS
    ], dtype=np.float32)
    
    FOCAL_LENGTH = iw
    CENTER = (iw/2, ih/2)
    CAMERA_MATRIX = np.array([[FOCAL_LENGTH, 0, CENTER[0]], [0, FOCAL_LENGTH, CENTER[1]], [0, 0, 1]], dtype=np.float32)
    DIST_COEFFS = np.zeros((4, 1))
    
    success, rotation_vector, _ = cv2.solvePnP(MEDIAPIPE_MODEL_POINTS, image_points, CAMERA_MATRIX, DIST_COEFFS)
    
    if not success:
        return None
        
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix)
    yaw = -yaw # Adjust sign

    face_x = image_points[0][0] # Nose tip X
    face_y = image_points[0][1] # Nose tip Y

    return {"yaw": yaw, "pitch": pitch, "roll": roll, "face": {"x": face_x, "y": face_y}}


# MAIN WEBCAM LOOP 

print("Running Full Controller (Mouse is ALWAYS ON, 'q' to quit)")
cap = cv2.VideoCapture(0)
cv2.namedWindow("Head Pose Mouse Controller")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    start_time = time.time()
    
    # Flip the frame
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB *once* for all MediaPipe models
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    status_text = ""
    current_time = time.time()
    
    # GESTURE RECOGNITION (Click & Scroll) 
    hand_results = hands.process(img_rgb)
    gesture_detected = False # Flag to stop mouse movement
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get all finger landmarks
            landmarks = hand_landmarks.landmark
            
            # NEW: Fist Detection Logic
            # A fist is when all 4 fingertips are below their middle knuckle
            is_fist = (
                landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
                landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y and
                landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_PIP].y
            )

            #  Pinch Detection Logic 
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pinch_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
            is_pinch = pinch_distance < 0.05
            
            #  Event Handling 
            if is_fist:
                gesture_detected = True
                status_text = "Fist"
                if (current_time - last_click_time) > CLICK_COOLDOWN:
                    mouse_x, mouse_y = pyautogui.position()
                    if mouse_y > 50: # Check if cursor is not in the title bar
                        pyautogui.click()
                        status_text = "CLICK!"
                        print("clicked")
                        last_click_time = current_time
                        
                    else:
                        status_text = "Click (Safe Zone)"
                        print("Click blocked by Safe Zone")
            
            elif is_pinch:
                gesture_detected = True
                index_tip_y = index_tip.y  
                if prev_y is not None:
                    if index_tip_y < prev_y - 0.01:
                        pyautogui.scroll(20) # Scroll up
                        status_text = "Scrolling UP"
                    elif index_tip_y > prev_y + 0.01:  
                        pyautogui.scroll(-20) # Scroll down
                        status_text = "Scrolling DOWN"
                prev_y = index_tip_y
            else:
                prev_y = None # Reset scroll if no pinch
    else:
        prev_y = None # Reset scroll if no hand
        
    #  Head-Pose-to-Move (MediaPipe Face)=
    gaze_data = estimate_head_pose_mediapipe(img_rgb)
    
    # Only move mouse if face is detected AND no gesture is being performed
    if gaze_data and not gesture_detected:
        # Smoothing
        if smoothed_yaw is None:
            smoothed_yaw = gaze_data["yaw"]
            smoothed_pitch = gaze_data["pitch"]
        else:
            smoothed_yaw = alpha * gaze_data["yaw"] + (1 - alpha) * smoothed_yaw
            smoothed_pitch = alpha * gaze_data["pitch"] + (1 - alpha) * smoothed_pitch
            
        gaze_data["yaw"] = smoothed_yaw
        gaze_data["pitch"] = smoothed_pitch
        
        if status_text == "":
            status_text = "Moving"
        
        screen_width, screen_height = pyautogui.size()

        # Tuned ranges for responsiveness
        YAW_RANGE   = [-15, 15]  # [Min angle Left, Max angle Right]
        PITCH_RANGE = [-10, 10]  # [Min angle Down, Max angle Up]
        
        #  Flipped the output range for target_x to invert horizontal controls
        target_x = np.interp(gaze_data["yaw"], YAW_RANGE, [screen_width - 1, 0])
        target_y = np.interp(gaze_data["pitch"], PITCH_RANGE, [screen_height - 1, 0])
        
        target_x = np.clip(target_x, 0, screen_width - 1)
        target_y = np.clip(target_y, 0, screen_height - 1)
            
        pyautogui.moveTo(target_x, target_y)
    
    # Calculate FPS
    processing_time = time.time() - start_time
    fps = 1 / processing_time if processing_time > 0 else float('inf')
    
    # Draw all info
    draw_info(frame, gaze_data, fps, status_text)
    
    # Display the result
    cv2.imshow("Head Pose Mouse Controller", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

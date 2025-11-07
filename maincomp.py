import cv2
import numpy as np
import dlib
import mediapipe as mp
import pyautogui
import time 

# --- 0. CONFIGURATION ---
# Set this to True to use MediaPipe, False to use Dlib
USE_MEDIAPIPE = False

# --- 1. DLIB SETUP ---
print("Loading Dlib model...")
try:
    dlib_detector = dlib.get_frontal_face_detector()
    dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except RuntimeError:
    print("ERROR: Could not find 'shape_predictor_68_face_landmarks.dat'")
    print("Download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    print("And unzip it in the same folder as this script.")
    exit()

# Dlib 68-point model indices
DLIB_LANDMARK_IDS = [
    30,  # Nose Tip
    36,  # Left Eye Outer Corner
    45,  # Right Eye Outer Corner
    48,  # Left Mouth Corner
    54,  # Right Mouth Corner
    8    # Chin
]

# 3D model points for Dlib
DLIB_MODEL_POINTS = np.array([
    ( 0.0,    0.0,    0.0),   # Nose Tip
    (-22.0, -17.0, -26.0),   # Left Eye Outer Corner
    ( 22.0, -17.0, -26.0),   # Right Eye Outer Corner
    (-15.0,  27.0, -20.0),   # Left Mouth Corner
    ( 15.0,  27.0, -20.0),   # Right Mouth Corner
    ( 0.0,   60.0, -35.0)    # Chin
], dtype=np.float32)


# --- 2. MEDIAPIPE SETUP ---
print("Loading MediaPipe model...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# MediaPipe 6-point model indices
MEDIAPIPE_LANDMARK_IDS = [
    1,   # Nose Tip
    133, # Left Eye Outer Corner
    362, # Right Eye Outer Corner
    234, # Left Ear
    454, # Right Ear
    152  # Chin
]

# 3D model points for MediaPipe
MEDIAPIPE_MODEL_POINTS = np.array([
    ( 0.0,    0.0,    0.0),    # Nose Tip
    (-30.0, -30.0,  -30.0),    # Left Eye
    ( 30.0, -30.0,  -30.0),    # Right Eye
    (-60.0,  60.0,  -60.0),    # Left Ear
    ( 60.0,  60.0,  -60.0),    # Right Ear
    ( 0.0,  100.0, -100.0)    # Chin
], dtype=np.float32)


# --- 3. SHARED PARAMETERS & FUNCTIONS ---

# Smoothing parameters
alpha = 0.7 
smoothed_yaw = None
smoothed_pitch = None

# Camera Parameters (will be set in functions)
FOCAL_LENGTH = 1
CENTER = (0, 0)

def rotation_matrix_to_euler_angles(R):
    """ Converts a rotation matrix to Euler angles (yaw, pitch, roll) """
    yaw = np.arctan2(R[1, 0], R[0, 0]) * (180.0 / np.pi)
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)) * (180.0 / np.pi) 
    roll = np.arctan2(R[2, 1], R[2, 2]) * (180.0 / np.pi)
    return yaw, pitch, roll

def draw_info(img, gaze, fps, framework_name):
    """ Draws the gaze arrow and info text on the frame """
    face = gaze["face"]
    
    arrow_length = img.shape[1] / 3
    dx = -arrow_length * np.sin(np.radians(gaze["yaw"]))
    dy = -arrow_length * np.sin(np.radians(gaze["pitch"]))
    
    cv2.arrowedLine(
        img,
        (int(face["x"]), int(face["y"])),
        (int(face["x"] + dx), int(face["y"] + dy)),
        (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.2
    )
        
    cv2.putText(img, f"Yaw: {gaze['yaw']:.2f}", (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(img, f"Pitch: {gaze['pitch']:.2f}", (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(img, f"Roll: {gaze['roll']:.2f}", (20, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    
    # Display the framework and FPS
    color = (0, 255, 0) if framework_name == "MediaPipe" else (0, 0, 255)
    cv2.putText(img, f"Framework: {framework_name}", (20, 120), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    cv2.putText(img, f"FPS: {fps:.2f}", (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

# --- 4. POSE ESTIMATION FUNCTIONS ---

def estimate_head_pose_dlib(image):
    """ Estimates head pose using Dlib """
    global FOCAL_LENGTH, CENTER
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = dlib_detector(gray)
    
    if not faces:
        return None
        
    shape = dlib_predictor(gray, faces[0])
    ih, iw = image.shape[:2]
    
    image_points = np.array([
        [shape.part(i).x, shape.part(i).y]
        for i in DLIB_LANDMARK_IDS
    ], dtype=np.float32)
    
    # Camera Intrinsics
    FOCAL_LENGTH = iw
    CENTER = (iw/2, ih/2)
    CAMERA_MATRIX = np.array([
        [FOCAL_LENGTH, 0, CENTER[0]],
        [0, FOCAL_LENGTH, CENTER[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    DIST_COEFFS = np.zeros((4, 1)) # No distortion
    
    # SolvePnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        DLIB_MODEL_POINTS, image_points, CAMERA_MATRIX, DIST_COEFFS
    )
    
    if not success:
        return None
        
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix)
    yaw = -yaw # Adjust sign
    
    # Get face center
    x_coords = [p[0] for p in image_points]
    y_coords = [p[1] for p in image_points]
    face_x = np.mean(x_coords)
    face_y = np.mean(y_coords)

    return {"yaw": yaw, "pitch": pitch, "roll": roll, "face": {"x": face_x, "y": face_y}}

def estimate_head_pose_mediapipe(image):
    """ Estimates head pose using MediaPipe """
    global FOCAL_LENGTH, CENTER
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    
    if not results.multi_face_landmarks:
        return None
        
    face_landmarks = results.multi_face_landmarks[0]
    ih, iw = image.shape[:2]
    
    # Get 2D image points from MediaPipe
    image_points = np.array([
        [face_landmarks.landmark[i].x * iw, face_landmarks.landmark[i].y * ih]
        for i in MEDIAPIPE_LANDMARK_IDS
    ], dtype=np.float32)
    
    # Camera Intrinsics
    FOCAL_LENGTH = iw
    CENTER = (iw/2, ih/2)
    CAMERA_MATRIX = np.array([
        [FOCAL_LENGTH, 0, CENTER[0]],
        [0, FOCAL_LENGTH, CENTER[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    DIST_COEFFS = np.zeros((4, 1)) # No distortion
    
    # SolvePnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MEDIAPIPE_MODEL_POINTS, image_points, CAMERA_MATRIX, DIST_COEFFS
    )
    
    if not success:
        return None
        
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix)
    yaw = -yaw # Adjust sign

    # Get face center (use nose tip)
    face_x = image_points[0][0]
    face_y = image_points[0][1]

    return {"yaw": yaw, "pitch": pitch, "roll": roll, "face": {"x": face_x, "y": face_y}}


# --- 5. MAIN WEBCAM LOOP ---

print("Running Head Pose Comparison...")
print(f"*** USING {'MEDIAPIPE' if USE_MEDIAPIPE else 'DLIB'} FOR POSE ESTIMATION ***")
cap = cv2.VideoCapture(0)
cv2.namedWindow("Head Pose Comparison")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # --- Gaze Estimation (SWITCHABLE) ---
    framework_name = ""
    gaze_data = None
    
    if USE_MEDIAPIPE:
        start_time = time.time()
        gaze_data = estimate_head_pose_mediapipe(frame)
        processing_time = time.time() - start_time
        framework_name = "MediaPipe"
    else:
        start_time = time.time()
        gaze_data = estimate_head_pose_dlib(frame)
        processing_time = time.time() - start_time
        framework_name = "Dlib"
    
    # Calculate FPS
    if processing_time > 0:
        fps = 1 / processing_time
    else:
        fps = float('inf')

    if gaze_data:
        # Smoothing
        if smoothed_yaw is None:
            smoothed_yaw = gaze_data["yaw"]
            smoothed_pitch = gaze_data["pitch"]
        else:
            smoothed_yaw = alpha * gaze_data["yaw"] + (1 - alpha) * smoothed_yaw
            smoothed_pitch = alpha * gaze_data["pitch"] + (1 - alpha) * smoothed_pitch
            
        gaze_data["yaw"] = smoothed_yaw
        gaze_data["pitch"] = smoothed_pitch
        
        # Draw info
        draw_info(frame, gaze_data, fps, framework_name)
        
        # --- PyAutoGUI Mouse Control ---
        screen_width, screen_height = pyautogui.size()
        
        # This mapping can be tuned
        target_x = screen_width * (gaze_data["yaw"] + 40) / 80
        target_y = screen_height * (gaze_data["pitch"] + 30) / 60
        
        target_x = np.clip(target_x, 0, screen_width - 1)
        target_y = np.clip(target_y, 0, screen_height - 1)
        
        pyautogui.moveTo(target_x, target_y)

    # Display the result
    cv2.imshow("Head Pose Comparison", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
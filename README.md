Vision-Based Mouse Controller

A high-performance, hands-free mouse controller that uses head pose for cursor movement and hand gestures for clicks and scrolling. Built with MediaPipe for robust, real-time performance.

Features

Head-Pose-to-Move: Control the cursor's (X, Y) position by turning your head (Yaw and Pitch).

Fist-to-Click: Make a fist to perform a left-click.

Pinch-to-Scroll: Pinch your thumb and index finger and move vertically to scroll up or down.

Core Technology

This system is built entirely on the MediaPipe framework, chosen for its superior performance over Dlib.

MediaPipe Face Mesh: Tracks 468 facial landmarks to solve for head pose (Yaw, Pitch, Roll) using a PnP algorithm.

MediaPipe Hands: Tracks 21 hand landmarks to detect 'Fist' and 'Pinch' gestures.

OpenCV: Handles all webcam video capture and real-time image processing.

PyAutoGUI: Translates pose and gestures into programmatic mouse and scroll events.

Performance: Why MediaPipe?

A comparative analysis showed MediaPipe to be objectively superior to Dlib for this real-time application. MediaPipe provides a 2.5x higher framerate (avg. ~101.1 FPS vs. ~39.6 FPS) and, most importantly, does not fail when the head is turned.

Dlib's "frontal-only" detector fails at slight angles (~20° Yaw), while MediaPipe's model successfully tracks the full range of motion (~355° Yaw).

Setup and Installation

Clone the repository:

git clone [https://github.com/BlackBetty-SaysHello/Mouse-Detection-using-Head-Pose-.git](https://github.com/BlackBetty-SaysHello/Mouse-Detection-using-Head-Pose-.git)
cd Mouse-Detection-using-Head-Pose-

Run the main controller script 

headpose_mouse_control.py



Press 'q' while the webcam window is active to quit the application.

import cv2
import mediapipe as mp
import time
import numpy as np

kacamata = cv2.imread('kacamata.png', cv2.IMREAD_UNCHANGED)

# MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def overlay_transparent(background, overlay, x, y):
    bh, bw = background.shape[:2]
    h, w = overlay.shape[:2]

    if x + w > bw: w = bw - x
    if y + h > bh: h = bh - y
    if x < 0: overlay = overlay[:, -x:]; w += x; x = 0
    if y < 0: overlay = overlay[-y:, :]; h += y; y = 0

    if w <= 0 or h <= 0:
        return background

    overlay_crop = overlay[:h, :w]
    bg_crop = background[y:y+h, x:x+w]

    alpha = overlay_crop[:, :, 3:] / 255.0
    color = overlay_crop[:, :, :3]

    blended = alpha * color + (1 - alpha) * bg_crop
    background[y:y+h, x:x+w] = blended.astype(np.uint8)

    return background

def main():
    # Initialize webcam (latip 1)
    # cap = cv2.VideoCapture(1)
    
    # Initialize webcam (Eden 0)
    cap = cv2.VideoCapture(0)
    
    # Set resolution to 1920x1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Check if webcam opened successfully and resolution was set
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    
    # Initialize time for FPS calculation
    prev_time = time.time()
    
    # Initialize MediaPipe Face Detection
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break  # Ganti continue dengan break jika frame kosong
            
            image = cv2.flip(image, 1)  # agar tidak mirror

            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect faces
            results = face_detection.process(image_rgb)
            
            # Display FPS
            current_time = time.time()
            time_diff = current_time - prev_time
            if time_diff > 0:
                fps = 1 / time_diff
                cv2.putText(image, f"FPS: {int(fps)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            prev_time = current_time
            
            # Draw detection results
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
                    
                    # Get eye coordinates
                    keypoints = detection.location_data.relative_keypoints
                    left_eye = keypoints[0]
                    right_eye = keypoints[1]

                    # Convert to pixel coordinates
                    left_eye_x = int(left_eye.x * actual_width)
                    left_eye_y = int(left_eye.y * actual_height)
                    right_eye_x = int(right_eye.x * actual_width)
                    right_eye_y = int(right_eye.y * actual_height)

                    # Calculate center point and width between eyes
                    eye_center_x = (left_eye_x + right_eye_x) // 2
                    eye_center_y = (left_eye_y + right_eye_y) // 2
                    eye_width = int(2.5 * abs(right_eye_x - left_eye_x))  # Make it slightly wider

                    # Resize glasses image
                    scale_factor = eye_width / kacamata.shape[1]
                    new_glasses = cv2.resize(kacamata, (0, 0), fx=scale_factor, fy=scale_factor)

                    gh, gw = new_glasses.shape[:2]
                    top_left_x = eye_center_x - gw // 2
                    top_left_y = eye_center_y - gh // 2

                    # Overlay glasses
                    image = overlay_transparent(image, new_glasses, top_left_x, top_left_y)
            
            # Draw horizontal barriers
            # Calculate barrier positions - slightly toward center from top and bottom
            top_barrier_y = int(actual_height * 0.35)  # 35% from the top
            bottom_barrier_y = int(actual_height * 0.65)  # 65% from the top (25% from bottom)
            
            # Draw the barriers
            barrier_color = (67, 67, 84) # Brown (RGB: 84, 67, 67)
            barrier_thickness = 3
            
            # Top barrier
            cv2.line(image, (0, top_barrier_y), (actual_width, top_barrier_y), 
                     barrier_color, barrier_thickness)
            
            
            # Bottom barrier
            cv2.line(image, (0, bottom_barrier_y), (actual_width, bottom_barrier_y), 
                     barrier_color, barrier_thickness)
            
                    
            # Show webcam feed
            cv2.imshow('MediaPipe Face Detection', image)
            
            # Exit on ESC key press or when window is closed
            key = cv2.waitKey(5)
            if key == 27 or cv2.getWindowProperty('MediaPipe Face Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
                
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
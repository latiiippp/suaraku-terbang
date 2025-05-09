import cv2
import mediapipe as mp
import time

# MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(1)
    
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
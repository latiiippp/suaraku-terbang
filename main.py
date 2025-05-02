import cv2
import mediapipe as mp
import time

# MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set resolution to 1920x1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Check if webcam opened successfully and resolution was set
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get actual resolution (may differ from requested if camera doesn't support it)
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
                continue
            
            image = cv2.flip(image, 1)  # agar tidak mirror

            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect faces
            results = face_detection.process(image_rgb)
            
            # Display FPS
            current_time = time.time()
            time_diff = current_time - prev_time
            if time_diff > 0:  # Avoid division by zero
                fps = 1 / time_diff
                cv2.putText(image, f"FPS: {int(fps)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            prev_time = current_time
            
            # Draw detection results
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
                    
            # Show webcam feed
            cv2.imshow('MediaPipe Face Detection', image)
            
            # Exit on 'q' key press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

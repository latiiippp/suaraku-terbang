import cv2
import mediapipe as mp
import time
import numpy as np
import sounddevice as sd
import threading

kacamata = cv2.imread('assets/kacamata.png', cv2.IMREAD_UNCHANGED)
ball = cv2.imread('assets/ball.png', cv2.IMREAD_UNCHANGED)

# If the ball doesn't have an alpha channel, create one
if ball.shape[2] == 3:
    ball_with_alpha = np.zeros((ball.shape[0], ball.shape[1], 4), dtype=np.uint8)
    ball_with_alpha[:,:,:3] = ball
    ball_with_alpha[:,:,3] = 255
    ball = ball_with_alpha

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

sound_direction = "neutral"
last_movement_direction = "neutral"  # Tambahan untuk menyimpan arah gerakan terakhir
sound_info = {"bass_energy": 0, "treble_energy": 0, "dominant_freq": 0}

def detect_sound_direction(duration=0.1, sample_rate=44100):
    global sound_info, last_movement_direction
    # Ambil data audio selama durasi tertentu
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    audio_data = recording.flatten()

    # FFT untuk mendapatkan spektrum frekuensi
    fft = np.fft.fft(audio_data)
    freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
    magnitudes = np.abs(fft)

    # Ambil hanya bagian positif dari frekuensi (tanpa duplikasi)
    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitudes = magnitudes[:len(magnitudes)//2]

    # Temukan frekuensi dominan
    max_magnitude_idx = np.argmax(positive_magnitudes)
    dominant_freq = positive_freqs[max_magnitude_idx]
    
    # Hitung energi di frekuensi rendah dan tinggi untuk display
    bass_energy = np.sum(positive_magnitudes[(positive_freqs >= 20) & (positive_freqs <= 200)])
    treble_energy = np.sum(positive_magnitudes[(positive_freqs >= 500) & (positive_freqs <= 8000)])
    
    # Update sound info
    sound_info["bass_energy"] = bass_energy
    sound_info["treble_energy"] = treble_energy
    sound_info["dominant_freq"] = dominant_freq
    
    # Threshold untuk mendeteksi suara yang cukup keras
    energy_threshold = np.max(positive_magnitudes) * 0.05
    
    # Jika energi total terlalu rendah, anggap tidak ada suara
    if np.max(positive_magnitudes) < energy_threshold:
        return last_movement_direction  # Tetap gunakan arah gerakan terakhir

    # Logika sederhana berdasarkan frekuensi dominan
    if dominant_freq < 100:
        last_movement_direction = "down"  # Update arah gerakan terakhir
        return "down"  # Frekuensi rendah -> bola turun
    elif dominant_freq > 200:
        last_movement_direction = "up"    # Update arah gerakan terakhir
        return "up"    # Frekuensi tinggi -> bola naik
    else:
        return last_movement_direction  # Tetap gunakan arah gerakan terakhir

def sound_thread():
    global sound_direction
    while True:
        sound_direction = detect_sound_direction()

def main():
    cap = cv2.VideoCapture(1)
    
    # Set resolution to 1920x1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
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
    
    # Resize ball to desired size (adjust as needed)
    ball_width = 25 
    ball_aspect_ratio = ball.shape[0] / ball.shape[1]
    ball_height = int(ball_width * ball_aspect_ratio)
    resized_ball = cv2.resize(ball, (ball_width, ball_height))
    
    center_x = actual_width // 2 - resized_ball.shape[1] // 2
    center_y = actual_height // 2 - resized_ball.shape[0] // 2

    threading.Thread(target = sound_thread, daemon = True).start()

    frame_count = 0
    
    # Initialize MediaPipe Face Detection
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            # Deteksi suara
            direction = sound_direction

            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break 
            
            image = cv2.flip(image, 1)

            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect faces
            results = None
            if frame_count %1 == 0:
                results = face_detection.process(image_rgb)
            frame_count += 1
            
            # Display FPS
            current_time = time.time()
            time_diff = current_time - prev_time
            if time_diff > 0:
                fps = 1 / time_diff
                cv2.putText(image, f"FPS: {int(fps)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            prev_time = current_time
            
            # Display sound frequency information
            cv2.putText(image, f"Dominant Freq: {int(sound_info['dominant_freq'])} Hz", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(image, f"Bass Energy: {int(sound_info['bass_energy'])}", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image, f"Treble Energy: {int(sound_info['treble_energy'])}", 
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(image, f"Direction: {sound_direction}", 
                       (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Last Movement: {last_movement_direction}", 
                       (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

            # Draw detection results
            if results and results.detections:
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
                    eye_width = int(2.5 * abs(right_eye_x - left_eye_x))

                    # Resize glasses image
                    scale_factor = eye_width / kacamata.shape[1]
                    new_glasses = cv2.resize(kacamata, (0, 0), fx=scale_factor, fy=scale_factor)

                    gh, gw = new_glasses.shape[:2]
                    top_left_x = eye_center_x - gw // 2
                    top_left_y = eye_center_y - gh // 2

                    # Overlay glasses
                    image = overlay_transparent(image, new_glasses, top_left_x, top_left_y)
            
            direction = sound_direction
            # Gerakkan bola naik atau turun
            if direction == "down":
                center_y += 4
            elif direction == "up":
                center_y -= 4               
            
            # Ensure the ball stays within the frame
            center_y = max(0, min(center_y, actual_height - resized_ball.shape[0]))
            
            # Overlay the ball at the center position
            image = overlay_transparent(image, resized_ball, center_x, center_y)
            
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
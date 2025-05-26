import cv2
import mediapipe as mp
import time
import numpy as np
import sounddevice as sd
import threading

print(sd.query_devices())

kacamata = cv2.imread('assets/kacamata.png', cv2.IMREAD_UNCHANGED)
ball = cv2.imread('assets/ball.png', cv2.IMREAD_UNCHANGED)

# Placeholder if assets are not loaded
if kacamata is None:
    print("Warning: Could not load kacamata.png. Glasses overlay will not work.")
if ball is None:
    print("Error: Could not load ball.png. Creating a placeholder.")
    ball = np.zeros((50, 50, 4), dtype=np.uint8) 
    cv2.circle(ball, (25, 25), 20, (0,0,255,255), -1) # Red ball placeholder
elif ball.shape[2] == 3: # If ball is loaded but 3-channel, add alpha
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
    if x < 0: 
        overlay = overlay[:, -x:]
        w += x
        x = 0
    if y < 0: 
        overlay = overlay[-y:, :]
        h += y
        y = 0

    if w <= 0 or h <= 0 or overlay is None: # Added check for overlay is None
        return background

    overlay_crop = overlay[:h, :w]
    
    # Ensure overlay_crop has 4 channels if it's valid
    if overlay_crop.shape[2] < 4:
        # This might happen if the overlay image itself is not RGBA
        # For simplicity, we'll skip overlaying if it's not RGBA as expected
        print("Warning: Overlay image is not RGBA. Skipping overlay.")
        return background

    bg_region = background[y:y+h, x:x+w]
    
    # Ensure bg_region and overlay_crop are compatible for blending
    if bg_region.shape[0] != overlay_crop.shape[0] or bg_region.shape[1] != overlay_crop.shape[1]:
        # This can happen due to rounding or extreme cropping, skip if dimensions don't match
        return background

    alpha_overlay = overlay_crop[:, :, 3:] / 255.0
    color_overlay = overlay_crop[:, :, :3]

    # Standard alpha blending
    # C = C_overlay * alpha_overlay + C_background * (1 - alpha_overlay)
    
    blended_color = color_overlay * alpha_overlay + bg_region[:,:,:3] * (1.0 - alpha_overlay)
    
    background[y:y+h, x:x+w, :3] = blended_color.astype(np.uint8)

    # If background has an alpha channel, update it based on overlay's alpha
    if background.shape[2] == 4:
        alpha_bg_region = bg_region[:, :, 3:] / 255.0
        # New alpha: alpha_overlay + alpha_bg * (1 - alpha_overlay)
        new_alpha_bg = alpha_overlay + alpha_bg_region * (1.0 - alpha_overlay)
        background[y:y+h, x:x+w, 3] = (new_alpha_bg * 255).astype(np.uint8)

    return background

sound_direction = "neutral"
last_movement_direction = "neutral"
sound_info = {"bass_energy": 0, "treble_energy": 0, "dominant_freq": 0}

def detect_sound_direction(duration=0.1, sample_rate=44100):
    global sound_info, last_movement_direction
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32', device=31)
    sd.wait()
    audio_data = recording.flatten()

    fft = np.fft.fft(audio_data)
    freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
    magnitudes = np.abs(fft)

    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitudes = magnitudes[:len(magnitudes)//2]

    current_determined_direction = "neutral" # Default

    if len(positive_magnitudes) == 0 or np.sum(positive_magnitudes) == 0:
        sound_info["bass_energy"] = 0
        sound_info["treble_energy"] = 0
        sound_info["dominant_freq"] = 0
        # current_determined_direction remains "neutral"
    else:
        max_magnitude_idx = np.argmax(positive_magnitudes)
        dominant_freq = positive_freqs[max_magnitude_idx]
        
        # Update sound_info for display based on new frequency ranges
        sound_info["bass_energy"] = np.sum(positive_magnitudes[(positive_freqs >= 1) & (positive_freqs <= 150)])
        sound_info["treble_energy"] = np.sum(positive_magnitudes[(positive_freqs > 150)])
        sound_info["dominant_freq"] = dominant_freq

        if dominant_freq == 0:
            current_determined_direction = "neutral"
        elif 1 <= dominant_freq <= 150: # Bass
            current_determined_direction = "down"
        elif dominant_freq > 150: # Treble
            current_determined_direction = "up"
        # else, it remains "neutral" (e.g., if dominant_freq is somehow negative, though unlikely)
            
    last_movement_direction = current_determined_direction
    return current_determined_direction

def sound_thread():
    global sound_direction
    while True:
        sound_direction = detect_sound_direction()
        time.sleep(0.01) 

def main():
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    
    prev_time = time.time()
    
    ball_width = 25 
    if ball is None or ball.shape[1] == 0: # Should be handled by placeholder logic already
        ball_aspect_ratio = 1 
        # This block might not be strictly necessary if placeholder is guaranteed
        temp_ball_img = np.zeros((50, 50, 4), dtype=np.uint8)
        cv2.circle(temp_ball_img, (25, 25), 20, (0,0,255,255), -1)
        ball_height = int(ball_width * ball_aspect_ratio)
        resized_ball = cv2.resize(temp_ball_img, (ball_width, ball_height))
    else:
        ball_aspect_ratio = ball.shape[0] / ball.shape[1]
        ball_height = int(ball_width * ball_aspect_ratio)
        resized_ball = cv2.resize(ball, (ball_width, ball_height))
    
    # Initial ball position: far left, middle vertically
    center_x = 0 
    center_y = actual_height // 2 - resized_ball.shape[0] // 2

    threading.Thread(target=sound_thread, daemon=True).start()

    frame_count = 0
    
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break 
            
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = None
            if frame_count % 1 == 0:
                results = face_detection.process(image_rgb)
            frame_count += 1
            
            current_time = time.time()
            time_diff = current_time - prev_time
            if time_diff > 0:
                fps = 1 / time_diff
                cv2.putText(image, f"FPS: {int(fps)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            prev_time = current_time
            
            cv2.putText(image, f"Dominant Freq: {int(sound_info['dominant_freq'])} Hz", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(image, f"Bass Energy (1-150Hz): {int(sound_info['bass_energy'])}", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image, f"Treble Energy (>150Hz): {int(sound_info['treble_energy'])}", 
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(image, f"Direction: {sound_direction}", 
                       (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Last Movement: {last_movement_direction}", 
                       (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

            if results and results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
                    
                    if kacamata is not None and kacamata.shape[1] > 0:
                        keypoints = detection.location_data.relative_keypoints
                        left_eye = keypoints[0]
                        right_eye = keypoints[1]

                        left_eye_x = int(left_eye.x * actual_width)
                        left_eye_y = int(left_eye.y * actual_height)
                        right_eye_x = int(right_eye.x * actual_width)
                        right_eye_y = int(right_eye.y * actual_height)

                        eye_center_x = (left_eye_x + right_eye_x) // 2
                        eye_center_y = (left_eye_y + right_eye_y) // 2
                        eye_width = int(2.5 * abs(right_eye_x - left_eye_x))

                        scale_factor = eye_width / kacamata.shape[1]
                        new_glasses = cv2.resize(kacamata, (0, 0), fx=scale_factor, fy=scale_factor)

                        gh, gw = new_glasses.shape[:2]
                        top_left_x = eye_center_x - gw // 2
                        top_left_y = eye_center_y - gh // 2
                        image = overlay_transparent(image, new_glasses, top_left_x, top_left_y)
            
            current_sound_direction = sound_direction 
            
            ball_speed_vertical = 4
            ball_speed_horizontal = 2 

            if current_sound_direction == "down": # Freq 1-150 Hz
                center_y += ball_speed_vertical
                center_x += ball_speed_horizontal 
            elif current_sound_direction == "up": # Freq > 150 Hz
                center_y -= ball_speed_vertical
                center_x += ball_speed_horizontal
            # Jika current_sound_direction == "neutral" (Freq == 0 Hz), bola tetap diam
            
            # Batasan pergerakan bola
            center_y = max(0, min(center_y, actual_height - resized_ball.shape[0]))
            center_x = max(0, min(center_x, actual_width - resized_ball.shape[1])) 
            
            image = overlay_transparent(image, resized_ball, center_x, center_y)
            
            top_barrier_y = int(actual_height * 0.35)
            bottom_barrier_y = int(actual_height * 0.65)
            barrier_color = (67, 67, 84) 
            barrier_thickness = 3
            
            cv2.line(image, (0, top_barrier_y), (actual_width, top_barrier_y), 
                     barrier_color, barrier_thickness)
            cv2.line(image, (0, bottom_barrier_y), (actual_width, bottom_barrier_y), 
                     barrier_color, barrier_thickness)
            
            cv2.imshow('MediaPipe Face Detection', image)
            
            key = cv2.waitKey(5) & 0xFF 
            if key == 27 or cv2.getWindowProperty('MediaPipe Face Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
                
    cap.release()
    cv2.destroyAllWindows()
    sd.stop()

if __name__ == "__main__":
    main()
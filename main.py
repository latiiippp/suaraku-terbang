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
        overlay = overlay[:, -x:] # Crop overlay from left
        w += x # Adjust width
        x = 0
    if y < 0: 
        overlay = overlay[-y:, :] # Crop overlay from top
        h += y # Adjust height
        y = 0

    if w <= 0 or h <= 0 or overlay is None:
        return background

    overlay_crop = overlay[:h, :w]
    
    if overlay_crop.shape[2] < 4:
        print("Warning: Overlay image is not RGBA. Skipping overlay.")
        return background

    bg_region = background[y:y+h, x:x+w]
    
    if bg_region.shape[0] != overlay_crop.shape[0] or bg_region.shape[1] != overlay_crop.shape[1]:
        return background

    alpha_overlay = overlay_crop[:, :, 3:] / 255.0
    color_overlay = overlay_crop[:, :, :3]
    
    blended_color = color_overlay * alpha_overlay + bg_region[:,:,:3] * (1.0 - alpha_overlay)
    
    background[y:y+h, x:x+w, :3] = blended_color.astype(np.uint8)

    if background.shape[2] == 4: # If background itself has an alpha channel
        alpha_bg_region = bg_region[:, :, 3:] / 255.0
        new_alpha_bg = alpha_overlay + alpha_bg_region * (1.0 - alpha_overlay)
        background[y:y+h, x:x+w, 3] = (new_alpha_bg * 255).astype(np.uint8)

    return background

sound_direction = "neutral"
last_movement_direction = "neutral"
sound_info = {"bass_energy": 0, "treble_energy": 0, "dominant_freq": 0}

# Level settings
current_level = 1
max_level = 5
# Scoring system
current_score = 100
score_per_level = 50
barrier_penalty = 10
game_over = False
game_started = False

# UI Animation variables
collision_flash = 0
level_up_flash = 0
ui_animations = {"score_pulse": 0, "level_glow": 0, "background_wave": 0, "particle_time": 0}

# Particle system
particles = []
ball_trail = []
max_trail_length = 15

# Color schemes
color_schemes = {
    "primary": (100, 150, 255),
    "secondary": (50, 200, 150),
    "accent": (255, 100, 150),
    "background_dark": (15, 15, 25),
    "background_light": (25, 25, 35),
    "text_primary": (255, 255, 255),
    "text_secondary": (180, 180, 200),
    "success": (100, 255, 100),
    "warning": (255, 200, 100),
    "danger": (255, 100, 100)
}

# (top_factor, bottom_factor) for barrier y-positions relative to screen height
level_barrier_settings = {
    1: (0.30, 0.70),  # Gap: 40% of height
    2: (0.35, 0.65),  # Gap: 30% of height
    3: (0.40, 0.60),  # Gap: 20% of height
    4: (0.425, 0.575), # Gap: 15% of height
    5: (0.45, 0.55)   # Gap: 10% of height
}

def detect_sound_direction(duration=0.1, sample_rate=44100):
    global sound_info, last_movement_direction
    try:
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32', device=1)
        sd.wait()
        audio_data = recording.flatten()

        fft = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
        magnitudes = np.abs(fft)

        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitudes = magnitudes[:len(magnitudes)//2]

        current_determined_direction = "neutral" 

        if len(positive_magnitudes) == 0 or np.sum(positive_magnitudes) == 0:
            sound_info["bass_energy"] = 0
            sound_info["treble_energy"] = 0
            sound_info["dominant_freq"] = 0
        else:
            max_magnitude_idx = np.argmax(positive_magnitudes)
            dominant_freq = positive_freqs[max_magnitude_idx]
            
            sound_info["bass_energy"] = np.sum(positive_magnitudes[(positive_freqs >= 1) & (positive_freqs <= 150)])
            sound_info["treble_energy"] = np.sum(positive_magnitudes[(positive_freqs > 150)])
            sound_info["dominant_freq"] = dominant_freq

            if dominant_freq == 0:
                current_determined_direction = "neutral"
            elif 1 <= dominant_freq <= 150: 
                current_determined_direction = "down"
            elif dominant_freq > 150: 
                current_determined_direction = "up"
                
        last_movement_direction = current_determined_direction
        return current_determined_direction
    except Exception as e:
        print(f"Audio error: {e}")
        return "neutral"

def sound_thread():
    global sound_direction
    while True:
        sound_direction = detect_sound_direction()
        time.sleep(0.01) 

def check_collision_with_barriers(ball_x, ball_y, ball_width, ball_height, top_barrier_y, bottom_barrier_y, barrier_thickness):
    """Check if ball collides with top or bottom barriers"""
    ball_top = ball_y
    ball_bottom = ball_y + ball_height
    
    # Check collision with top barrier
    if ball_top <= top_barrier_y + barrier_thickness:
        return True
    
    # Check collision with bottom barrier  
    if ball_bottom >= bottom_barrier_y:
        return True
        
    return False

def draw_rounded_rectangle(img, pt1, pt2, color, thickness=2, radius=10, filled=False):
    """Draw a rounded rectangle"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    if filled:
        # Create mask for rounded rectangle
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x1+radius, y1), (x2-radius, y2), 255, -1)
        cv2.rectangle(mask, (x1, y1+radius), (x2, y2-radius), 255, -1)
        cv2.circle(mask, (x1+radius, y1+radius), radius, 255, -1)
        cv2.circle(mask, (x2-radius, y1+radius), radius, 255, -1)
        cv2.circle(mask, (x1+radius, y2-radius), radius, 255, -1)
        cv2.circle(mask, (x2-radius, y2-radius), radius, 255, -1)
        
        # Apply color to masked area
        img[mask == 255] = color
    else:
        # Draw outline
        cv2.line(img, (x1+radius, y1), (x2-radius, y1), color, thickness)
        cv2.line(img, (x1+radius, y2), (x2-radius, y2), color, thickness)
        cv2.line(img, (x1, y1+radius), (x1, y2-radius), color, thickness)
        cv2.line(img, (x2, y1+radius), (x2, y2-radius), color, thickness)
        cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness)

def draw_gradient_panel(img, pt1, pt2, color1, color2, alpha=0.8):
    """Draw a gradient panel with transparency"""
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img.copy()
    
    # Create gradient
    gradient = np.linspace(0, 1, y2-y1).reshape(-1, 1)
    gradient_rgb = gradient * np.array(color2) + (1-gradient) * np.array(color1)
    
    for i in range(y2-y1):
        color = tuple(map(int, gradient_rgb[i]))
        cv2.rectangle(overlay, (x1, y1+i), (x2, y1+i+1), color, -1)
    
    # Apply transparency
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def draw_modern_hud(img, width, height, fps, score, level, sound_info, sound_direction, collision_flash, level_up_flash):
    """Draw modern HUD with panels and animations"""
    global ui_animations
    
    # Update animations
    ui_animations["score_pulse"] = (ui_animations["score_pulse"] + 1) % 60
    ui_animations["level_glow"] = (ui_animations["level_glow"] + 1) % 120
    
    # Top panel - Score and Level
    panel_height = 80
    draw_gradient_panel(img, (0, 0), (width, panel_height), (30, 30, 40), (20, 20, 30), 0.85)
    
    # Score panel with pulse effect
    score_pulse = abs(np.sin(ui_animations["score_pulse"] * 0.1)) * 20
    score_color = (50 + int(score_pulse), 255, 50 + int(score_pulse))
    
    cv2.putText(img, "SCORE", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(img, f"{score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, score_color, 2)
    
    # Level panel with glow effect
    level_glow = abs(np.sin(ui_animations["level_glow"] * 0.05)) * 30
    level_color = (50, 150 + int(level_glow), 255)
    
    cv2.putText(img, "LEVEL", (width - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(img, f"{level}/{max_level}", (width - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, level_color, 2)
    
    # Progress bar for level
    bar_width = 100
    bar_height = 8
    bar_x = width - 130
    bar_y = 60
    
    # Background bar
    draw_rounded_rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), filled=True, radius=4)
    
    # Progress bar
    progress = level / max_level
    progress_width = int(bar_width * progress)
    if progress_width > 0:
        draw_rounded_rectangle(img, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), level_color, filled=True, radius=4)
    
    # Side panel - Audio info
    panel_width = 250
    panel_x = 10
    panel_y = 90
    panel_bottom = height - 20
    
    draw_gradient_panel(img, (panel_x, panel_y), (panel_x + panel_width, panel_bottom), (25, 25, 35), (15, 15, 25), 0.85)
    draw_rounded_rectangle(img, (panel_x, panel_y), (panel_x + panel_width, panel_bottom), (100, 100, 120), 2, 8)
    
    # Audio info with modern styling
    y_offset = panel_y + 25
    
    cv2.putText(img, "AUDIO ANALYSIS", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_offset += 30
    
    # Frequency display with color coding
    freq = int(sound_info['dominant_freq'])
    freq_color = (0, 255, 255) if freq > 150 else (255, 255, 0) if freq > 0 else (128, 128, 128)
    cv2.putText(img, f"Frequency: {freq} Hz", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, freq_color, 1)
    y_offset += 25
    
    # Bass energy bar
    bass_energy = min(int(sound_info['bass_energy'] / 100), 100)
    cv2.putText(img, "Bass:", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    bar_bg = (panel_x + 60, y_offset - 10, panel_x + 200, y_offset)
    cv2.rectangle(img, (bar_bg[0], bar_bg[1]), (bar_bg[2], bar_bg[3]), (50, 50, 50), -1)
    if bass_energy > 0:
        bar_fill_width = int((bar_bg[2] - bar_bg[0]) * bass_energy / 100)
        cv2.rectangle(img, (bar_bg[0], bar_bg[1]), (bar_bg[0] + bar_fill_width, bar_bg[3]), (0, 255, 255), -1)
    y_offset += 25
    
    # Treble energy bar
    treble_energy = min(int(sound_info['treble_energy'] / 100), 100)
    cv2.putText(img, "Treble:", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    bar_bg = (panel_x + 60, y_offset - 10, panel_x + 200, y_offset)
    cv2.rectangle(img, (bar_bg[0], bar_bg[1]), (bar_bg[2], bar_bg[3]), (50, 50, 50), -1)
    if treble_energy > 0:
        bar_fill_width = int((bar_bg[2] - bar_bg[0]) * treble_energy / 100)
        cv2.rectangle(img, (bar_bg[0], bar_bg[1]), (bar_bg[0] + bar_fill_width, bar_bg[3]), (255, 0, 255), -1)
    y_offset += 30
    
    # Direction indicator with arrows
    direction_colors = {
        "up": (0, 255, 0),
        "down": (255, 100, 0),
        "neutral": (128, 128, 128)
    }
    
    cv2.putText(img, "DIRECTION:", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    direction_color = direction_colors.get(sound_direction, (128, 128, 128))
    
    # Draw direction arrow
    arrow_center = (panel_x + 120, y_offset - 8)
    if sound_direction == "up":
        cv2.arrowedLine(img, (arrow_center[0], arrow_center[1] + 10), (arrow_center[0], arrow_center[1] - 10), direction_color, 3)
    elif sound_direction == "down":
        cv2.arrowedLine(img, (arrow_center[0], arrow_center[1] - 10), (arrow_center[0], arrow_center[1] + 10), direction_color, 3)
    else:
        cv2.circle(img, arrow_center, 5, direction_color, -1)
    
    cv2.putText(img, sound_direction.upper(), (panel_x + 140, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, direction_color, 1)
    
    # FPS display
    cv2.putText(img, f"FPS: {int(fps)}", (panel_x + 15, panel_bottom - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
    
    # Collision flash effect
    if collision_flash > 0:
        flash_alpha = collision_flash / 30.0
        flash_overlay = img.copy()
        cv2.rectangle(flash_overlay, (0, 0), (width, height), (0, 0, 255), -1)
        cv2.addWeighted(img, 1-flash_alpha*0.3, flash_overlay, flash_alpha*0.3, 0, img)
    
    # Level up flash effect
    if level_up_flash > 0:
        flash_alpha = level_up_flash / 60.0
        flash_overlay = img.copy()
        cv2.rectangle(flash_overlay, (0, 0), (width, height), (0, 255, 0), -1)
        cv2.addWeighted(img, 1-flash_alpha*0.2, flash_overlay, flash_alpha*0.2, 0, img)

def draw_modern_barriers(img, width, height, top_y, bottom_y, thickness):
    """Draw modern looking barriers with glow effect"""
    barrier_color = (100, 150, 255)
    glow_color = (50, 100, 200)
    
    # Draw glow effect
    for i in range(3):
        glow_thickness = thickness + (3-i) * 2
        alpha = 0.3 - i * 0.1
        overlay = img.copy()
        cv2.line(overlay, (0, top_y), (width, top_y), glow_color, glow_thickness)
        cv2.line(overlay, (0, bottom_y), (width, bottom_y), glow_color, glow_thickness)
        cv2.addWeighted(img, 1-alpha, overlay, alpha, 0, img)
    
    # Draw main barriers
    cv2.line(img, (0, top_y), (width, top_y), barrier_color, thickness)
    cv2.line(img, (0, bottom_y), (width, bottom_y), barrier_color, thickness)

def draw_modern_game_over(width, height, final_score, max_score_achieved):
    # game over
    game_over_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background gradient
    draw_gradient_panel(game_over_img, (0, 0), (width, height), (20, 20, 30), (40, 40, 60), 1.0)
    
    # Main panel
    panel_width = 400
    panel_height = 300
    panel_x = (width - panel_width) // 2
    panel_y = (height - panel_height) // 2
    
    draw_gradient_panel(game_over_img, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                       (60, 60, 80), (40, 40, 60), 0.95)
    draw_rounded_rectangle(game_over_img, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                          (150, 150, 200), 3, 15)
    
    # Game Over text with glow
    text_center_x = width // 2 - 120
    text_y = panel_y + 80
    
    # Glow effect for title
    cv2.putText(game_over_img, "GAME OVER", (text_center_x, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Score display
    score_y = text_y + 80
    cv2.putText(game_over_img, f"Final Score: {final_score}", (text_center_x - 20, score_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
    
    # Instructions
    instruction_y = score_y + 60
    cv2.putText(game_over_img, "Press any key to exit", (text_center_x - 40, instruction_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    return game_over_img

class Particle:
    def __init__(self, x, y, vx, vy, color, size, life):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.size = size
        self.life = life
        self.max_life = life
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # gravity
        self.life -= 1
        
    def is_alive(self):
        return self.life > 0
        
    def draw(self, img):
        if self.is_alive():
            alpha = self.life / self.max_life
            size = int(self.size * alpha)
            if size > 0:
                cv2.circle(img, (int(self.x), int(self.y)), size, self.color, -1)

def add_particles(x, y, color, count=10):
    global particles
    for _ in range(count):
        vx = np.random.uniform(-3, 3)
        vy = np.random.uniform(-5, -1)
        size = np.random.randint(2, 6)
        life = np.random.randint(20, 40)
        particles.append(Particle(x, y, vx, vy, color, size, life))

def update_particles():
    global particles
    particles = [p for p in particles if p.is_alive()]
    for particle in particles:
        particle.update()

def draw_particles(img):
    for particle in particles:
        particle.draw(img)

def add_ball_trail(x, y):
    global ball_trail
    ball_trail.append((x, y))
    if len(ball_trail) > max_trail_length:
        ball_trail.pop(0)

def draw_ball_trail(img):
    if len(ball_trail) < 2:
        return
    
    for i in range(1, len(ball_trail)):
        alpha = i / len(ball_trail)
        thickness = int(3 * alpha)
        if thickness > 0:
            color = tuple(int(c * alpha) for c in color_schemes["accent"])
            cv2.line(img, ball_trail[i-1], ball_trail[i], color, thickness)

def draw_glassmorphism_panel(img, pt1, pt2, blur_strength=15, alpha=0.3):
    """Create glassmorphism effect panel"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Ensure coordinates are valid and within bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    
    # Check if region is valid
    if x2 <= x1 or y2 <= y1:
        return
    
    # Ensure blur strength is odd and at least 1
    blur_strength = max(1, blur_strength)
    if blur_strength % 2 == 0:
        blur_strength += 1
    
    # Extract background region
    bg_region = img[y1:y2, x1:x2].copy()
    
    # Check if region is large enough for blur
    if bg_region.shape[0] < blur_strength or bg_region.shape[1] < blur_strength:
        # Use smaller blur for small regions
        blur_strength = min(bg_region.shape[0], bg_region.shape[1])
        if blur_strength < 3:
            blur_strength = 1
        elif blur_strength % 2 == 0:
            blur_strength -= 1
    
    # Apply blur only if region is valid
    if bg_region.shape[0] > 0 and bg_region.shape[1] > 0:
        blurred = cv2.GaussianBlur(bg_region, (blur_strength, blur_strength), 0)
        
        # Create white overlay for glass effect
        overlay = np.ones_like(blurred) * 40
        overlay[:, :, 0] = 60  # Slight blue tint
        overlay[:, :, 1] = 60
        overlay[:, :, 2] = 80
        
        # Blend with blurred background
        glass_effect = cv2.addWeighted(blurred, 1-alpha, overlay, alpha, 0)
        
        # Apply back to image
        img[y1:y2, x1:x2] = glass_effect
        
        # Add border glow
        border_color = color_schemes["primary"]
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 2)
        
        # Add inner glow
        inner_overlay = img.copy()
        cv2.rectangle(inner_overlay, (x1+2, y1+2), (x2-2, y2-2), border_color, 1)
        cv2.addWeighted(img, 0.9, inner_overlay, 0.1, 0, img)

def draw_animated_background(img, width, height):
    """Draw animated background with waves and patterns"""
    global ui_animations
    
    ui_animations["background_wave"] = (ui_animations["background_wave"] + 1) % 360
    
    # Create wave pattern
    wave_offset = ui_animations["background_wave"] * 0.02
    
    for y in range(0, height, 20):
        for x in range(0, width, 20):
            wave = np.sin((x * 0.01) + wave_offset) * np.cos((y * 0.01) + wave_offset)
            intensity = int(abs(wave) * 15)
            
            color = (
                color_schemes["background_dark"][0] + intensity,
                color_schemes["background_dark"][1] + intensity,
                color_schemes["background_dark"][2] + intensity
            )
            
            cv2.circle(img, (x, y), 2, color, -1)

def draw_sound_visualizer(img, x, y, width, height, sound_info):
    """Draw real-time sound visualization"""
    # Frequency bars
    bar_count = 20
    bar_width = width // bar_count
    
    # Simulate frequency spectrum
    bass_ratio = min(sound_info['bass_energy'] / 1000, 1.0)
    treble_ratio = min(sound_info['treble_energy'] / 1000, 1.0)
    
    for i in range(bar_count):
        # Create varying heights based on frequency
        if i < bar_count // 2:
            bar_height = int(height * bass_ratio * (0.5 + 0.5 * np.sin(i * 0.5 + time.time() * 2)))
        else:
            bar_height = int(height * treble_ratio * (0.5 + 0.5 * np.sin(i * 0.5 + time.time() * 3)))
        
        bar_x = x + i * bar_width
        bar_y = y + height - bar_height
        
        # Color gradient from bass (blue) to treble (red)
        ratio = i / bar_count
        color = (
            int(100 + 155 * ratio),
            int(150 - 100 * ratio),
            int(255 - 155 * ratio)
        )
        
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width - 2, y + height), color, -1)

def draw_frequency_display(img, x, y, width, height, sound_info):
    """Draw large frequency number display"""
    freq = int(sound_info['dominant_freq'])
    
    # Background panel
    panel_color = (30, 30, 50)
    cv2.rectangle(img, (x, y), (x + width, y + height), panel_color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), color_schemes["primary"], 2)
    
    # Title
    cv2.putText(img, "FREQUENCY", (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_schemes["text_secondary"], 2)
    
    # Large frequency number
    freq_text = f"{freq}"
    freq_color = color_schemes["warning"] if freq > 150 else color_schemes["secondary"] if freq > 0 else color_schemes["text_secondary"]
    
    # Glow effect for frequency
    for i in range(3):
        glow_alpha = 0.3 - i * 0.1
        glow_offset = i * 2
        overlay = img.copy()
        cv2.putText(overlay, freq_text, (x + 15 + glow_offset, y + 70 + glow_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, freq_color, 4)
        cv2.addWeighted(img, 1-glow_alpha, overlay, glow_alpha, 0, img)
    
    cv2.putText(img, freq_text, (x + 15, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 2.5, color_schemes["text_primary"], 3)
    cv2.putText(img, "Hz", (x + 15, y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_schemes["text_secondary"], 2)

def draw_game_info_panel(img, x, y, width, height, score, level, fps):
    """Draw game information panel"""
    # Background
    panel_color = (25, 25, 40)
    cv2.rectangle(img, (x, y), (x + width, y + height), panel_color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), color_schemes["primary"], 2)
    
    # Title
    cv2.putText(img, "GAME INFO", (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_schemes["text_primary"], 2)
    
    current_y = y + 50
    
    # Score section
    cv2.putText(img, "SCORE:", (x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_schemes["text_secondary"], 1)
    score_color = color_schemes["success"]
    cv2.putText(img, f"{score}", (x + 90, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2)
    current_y += 35
    
    # Level section
    cv2.putText(img, "LEVEL:", (x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_schemes["text_secondary"], 1)
    level_color = color_schemes["primary"]
    cv2.putText(img, f"{level}/{max_level}", (x + 90, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, level_color, 2)
    current_y += 35
    
    # Level progress bar
    bar_width = width - 20
    bar_height = 8
    bar_x = x + 10
    bar_y = current_y
    
    # Background bar
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)
    
    # Progress
    progress = level / max_level
    progress_width = int(bar_width * progress)
    if progress_width > 0:
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), level_color, -1)
    
    current_y += 35
    
    # FPS
    cv2.putText(img, "FPS:", (x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_schemes["text_secondary"], 1)
    fps_color = color_schemes["success"] if fps > 30 else color_schemes["warning"] if fps > 15 else color_schemes["danger"]
    cv2.putText(img, f"{int(fps)}", (x + 60, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

def draw_audio_control_panel(img, x, y, width, height, sound_info, sound_direction):
    """Draw audio control information panel"""
    # Background
    panel_color = (25, 25, 40)
    cv2.rectangle(img, (x, y), (x + width, y + height), panel_color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), color_schemes["secondary"], 2)
    
    # Title
    cv2.putText(img, "AUDIO CONTROL", (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_schemes["text_primary"], 2)
    
    current_y = y + 50
    
    # Direction indicator
    direction_colors = {
        "up": color_schemes["success"],
        "down": color_schemes["warning"],
        "neutral": color_schemes["text_secondary"]
    }
    
    cv2.putText(img, "DIRECTION:", (x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_schemes["text_secondary"], 1)
    current_color = direction_colors.get(sound_direction, color_schemes["text_secondary"])
    cv2.putText(img, sound_direction.upper(), (x + 10, current_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, current_color, 2)
    
    # Direction arrow
    arrow_center = (x + width - 40, current_y + 10)
    arrow_size = 15
    
    if sound_direction == "up":
        pts = np.array([
            [arrow_center[0], arrow_center[1] - arrow_size],
            [arrow_center[0] - arrow_size//2, arrow_center[1] + arrow_size//2],
            [arrow_center[0] + arrow_size//2, arrow_center[1] + arrow_size//2]
        ], np.int32)
        cv2.fillPoly(img, [pts], current_color)
    elif sound_direction == "down":
        pts = np.array([
            [arrow_center[0], arrow_center[1] + arrow_size],
            [arrow_center[0] - arrow_size//2, arrow_center[1] - arrow_size//2],
            [arrow_center[0] + arrow_size//2, arrow_center[1] - arrow_size//2]
        ], np.int32)
        cv2.fillPoly(img, [pts], current_color)
    else:
        cv2.circle(img, arrow_center, arrow_size//2, current_color, -1)
    
    current_y += 60
    
    # Audio energy bars
    cv2.putText(img, "BASS ENERGY:", (x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_schemes["text_secondary"], 1)
    current_y += 20
    
    # Bass bar
    bass_energy = min(int(sound_info['bass_energy'] / 50), 100)
    bar_width = width - 20
    bar_height = 8
    bar_x = x + 10
    bar_y = current_y
    
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    if bass_energy > 0:
        bass_fill_width = int(bar_width * bass_energy / 100)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bass_fill_width, bar_y + bar_height), color_schemes["secondary"], -1)
    
    current_y += 30
    
    # Treble energy
    cv2.putText(img, "TREBLE ENERGY:", (x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_schemes["text_secondary"], 1)
    current_y += 20
    
    # Treble bar
    treble_energy = min(int(sound_info['treble_energy'] / 50), 100)
    bar_y = current_y
    
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    if treble_energy > 0:
        treble_fill_width = int(bar_width * treble_energy / 100)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + treble_fill_width, bar_y + bar_height), color_schemes["warning"], -1)

def draw_split_interface(img, width, height, fps, score, level, sound_info, sound_direction, collision_flash, level_up_flash):
    """Draw split interface with game view and information panels"""
    # Define layout - make info panel narrower to show more of webcam
    info_panel_width = 250  # Reduced from 280
    game_view_width = width - info_panel_width
    
    # Don't draw game view background - keep webcam as background
    # Only draw info panel background (right side)
    info_bg_color = (15, 15, 25)
    cv2.rectangle(img, (game_view_width, 0), (width, height), info_bg_color, -1)
    
    # Separator line
    cv2.line(img, (game_view_width, 0), (game_view_width, height), color_schemes["primary"], 3)
    
    # Information panels on the right side - make more compact
    panel_x = game_view_width + 5  # Reduced from 10
    panel_width = info_panel_width - 10  # Reduced from 20
    
    # Frequency display (top) - make slightly smaller
    freq_height = 110  # Reduced from 120
    draw_frequency_display(img, panel_x, 10, panel_width, freq_height, sound_info)
    
    # Game info panel - make slightly smaller
    game_info_y = freq_height + 15  # Reduced from 20
    game_info_height = 160  # Reduced from 180
    draw_game_info_panel(img, panel_x, game_info_y, panel_width, game_info_height, score, level, fps)
    
    # Audio control panel
    audio_y = game_info_y + game_info_height + 10
    audio_height = height - audio_y - 10
    draw_audio_control_panel(img, panel_x, audio_y, panel_width, audio_height, sound_info, sound_direction)
    
    # Flash effects only on game view
    if collision_flash > 0:
        flash_intensity = collision_flash / 30.0
        flash_overlay = np.zeros((height, game_view_width, 3), dtype=np.uint8)
        flash_overlay[:] = color_schemes["danger"]
        cv2.addWeighted(img[:, :game_view_width], 1 - flash_intensity * 0.3, flash_overlay, flash_intensity * 0.3, 0, img[:, :game_view_width])
        
        # Add particles on collision
        if collision_flash == 30:
            add_particles(game_view_width//2, height//2, color_schemes["danger"], 20)
    
    if level_up_flash > 0:
        flash_intensity = level_up_flash / 60.0
        flash_overlay = np.zeros((height, game_view_width, 3), dtype=np.uint8)
        flash_overlay[:] = color_schemes["success"]
        cv2.addWeighted(img[:, :game_view_width], 1 - flash_intensity * 0.2, flash_overlay, flash_intensity * 0.2, 0, img[:, :game_view_width])
        
        # Add celebration particles
        if level_up_flash == 60:
            add_particles(game_view_width//2, 100, color_schemes["success"], 30)
    
    return game_view_width

def draw_game_view_overlay(img, game_view_width, height, current_level):
    """Draw minimal overlay on game view with semi-transparent background"""
    
    # Level indicator with background (top left of game view)
    level_text = f"LEVEL {current_level}"
    text_size = cv2.getTextSize(level_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    
    # Semi-transparent background for level text
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 5), (text_size[0] + 30, 40), (0, 0, 0), -1)
    cv2.addWeighted(img, 0.7, overlay, 0.3, 0, img)
    
    cv2.putText(img, level_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_schemes["text_primary"], 2)
    
    # Instructions with background (bottom of game view)
    instruction_y = height - 60
    instructions = [
        "Use your voice to control the ball",
        "High pitch = UP, Low pitch = DOWN"
    ]
    
    # Calculate background size for instructions
    max_text_width = 0
    for instruction in instructions:
        text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        max_text_width = max(max_text_width, text_size[0])
    
    # Semi-transparent background for instructions
    overlay = img.copy()
    cv2.rectangle(overlay, (10, instruction_y - 15), (max_text_width + 30, height - 10), (0, 0, 0), -1)
    cv2.addWeighted(img, 0.7, overlay, 0.3, 0, img)
    
    # Draw instructions
    for i, instruction in enumerate(instructions):
        cv2.putText(img, instruction, (20, instruction_y + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_schemes["text_secondary"], 1)

def draw_start_screen(width, height):
    """Create an attractive start screen"""
    start_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Animated background
    draw_animated_background(start_img, width, height)
    
    # Main panel
    panel_width = min(500, width - 40)  # Ensure panel fits in screen
    panel_height = min(400, height - 40)
    panel_x = (width - panel_width) // 2
    panel_y = (height - panel_height) // 2
    
    # Ensure panel dimensions are valid
    if panel_width > 10 and panel_height > 10:
        draw_glassmorphism_panel(start_img, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 15, 0.4)
    
    # Title with animated glow
    title_y = panel_y + 100
    title_text = "SUARAKU TERBANG"
    
    # Animated title glow
    glow_intensity = abs(np.sin(time.time() * 2))
    for i in range(5):
        glow_alpha = (0.5 - i * 0.1) * glow_intensity
        glow_offset = i * 3
        overlay = start_img.copy()
        cv2.putText(overlay, title_text, (panel_x + 50 + glow_offset, title_y + glow_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_schemes["primary"], 4)
        cv2.addWeighted(start_img, 1-glow_alpha, overlay, glow_alpha, 0, start_img)
    
    cv2.putText(start_img, title_text, (panel_x + 50, title_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_schemes["text_primary"], 3)
    
    # Instructions
    instructions = [
        "Voice-Controlled Ball Game",
        "",
        "How to play:",
        "• High pitch sounds = Move UP",
        "• Low pitch sounds = Move DOWN", 
        "• Navigate through barriers",
        "• Reach the end to level up!",
        "• Avoid hitting barriers to keep score",
        "",
        "Press SPACE to start"
    ]
    
    instruction_y = title_y + 60
    for instruction in instructions:
        if instruction:
            font_size = 0.7 if instruction == "Voice-Controlled Ball Game" else 0.5
            color = color_schemes["text_primary"] if instruction == "Voice-Controlled Ball Game" else color_schemes["text_secondary"]
            cv2.putText(start_img, instruction, (panel_x + 50, instruction_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1 if font_size == 0.5 else 2)
        instruction_y += 25
    
    return start_img

def main():
    global current_level, current_score, game_over, collision_flash, level_up_flash, game_started

    cap = cv2.VideoCapture(0)
    
    # Wider resolution to better accommodate webcam feed and info panel
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increased from 1024
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Increased from 600
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    
    # Get webcam's native aspect ratio to ensure proper display
    webcam_aspect_ratio = actual_width / actual_height
    
    # Calculate the ideal game view width based on the webcam's aspect ratio
    desired_game_view_height = actual_height
    desired_game_view_width = int(desired_game_view_height * webcam_aspect_ratio)
    
    # Total window width needs to accommodate both game view and info panel
    # MODIFIED: Adjusted info panel width for window_width calculation
    info_panel_width_for_calc = 80
    window_width = desired_game_view_width + info_panel_width_for_calc
    window_height = actual_height
    
    print(f"Window dimensions: {window_width}x{window_height}")
    
    prev_time = time.time()
    max_score_achieved = current_score
    
    ball_width = 25 
    if ball is None or ball.shape[1] == 0: 
        ball_aspect_ratio = 1 
        temp_ball_img = np.zeros((50, 50, 4), dtype=np.uint8)
        cv2.circle(temp_ball_img, (25, 25), 20, (0,0,255,255), -1)
        ball_height = int(ball_width * ball_aspect_ratio)
        resized_ball = cv2.resize(temp_ball_img, (ball_width, ball_height))
    else:
        ball_aspect_ratio = ball.shape[0] / ball.shape[1]
        ball_height = int(ball_width * ball_aspect_ratio)
        resized_ball = cv2.resize(ball, (ball_width, ball_height))
    
    center_x = 0 
    center_y = actual_height // 2 - resized_ball.shape[0] // 2

    threading.Thread(target=sound_thread, daemon=True).start()

    frame_count = 0
    # MODIFIED: collision_cooldown is not strictly necessary if game ends on first hit, but kept for flash
    collision_cooldown = 0 
    
    # Show start screen
    while not game_started:
        start_img = draw_start_screen(actual_width, actual_height)
        cv2.imshow('Suaraku Terbang', start_img)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            game_started = True
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return
    
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while cap.isOpened() and not game_over:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break 
            
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = None
            if frame_count % 1 == 0: # Process every frame for face detection
                results = face_detection.process(image_rgb)
            frame_count += 1
            
            current_time = time.time()
            time_diff = current_time - prev_time
            fps = 1 / time_diff if time_diff > 0 else 0
            prev_time = current_time
            
            max_score_achieved = max(max_score_achieved, current_score)

            if results and results.detections:
                for detection in results.detections:
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

                        if eye_width > 0: 
                            scale_factor = eye_width / kacamata.shape[1]
                            # MODIFIED: Ensure new_glasses has positive dimensions before overlay
                            if scale_factor > 0:
                                new_glasses_width = int(kacamata.shape[1] * scale_factor)
                                new_glasses_height = int(kacamata.shape[0] * scale_factor)
                                if new_glasses_width > 0 and new_glasses_height > 0:
                                    new_glasses = cv2.resize(kacamata, (new_glasses_width, new_glasses_height))
                                    gh, gw = new_glasses.shape[:2]
                                    top_left_x = eye_center_x - gw // 2
                                    top_left_y = eye_center_y - gh // 2
                                    image = overlay_transparent(image, new_glasses, top_left_x, top_left_y)
            
            frame_buffer = np.zeros((window_height, window_width, 3), dtype=np.uint8)
            
            if image.shape[0] != window_height or image.shape[1] != window_width:
                display_img = cv2.resize(image, (desired_game_view_width, desired_game_view_height))
                # Ensure display_img is copied correctly if its width is less than frame_buffer's width
                if display_img.shape[1] <= frame_buffer.shape[1] and display_img.shape[0] <= frame_buffer.shape[0]:
                     frame_buffer[:display_img.shape[0], :display_img.shape[1]] = display_img
                else: # Fallback if something is wrong with resize calculation
                    frame_buffer = cv2.resize(image, (window_width, window_height))

            else:
                frame_buffer = image.copy()
            
            game_view_width = draw_split_interface(frame_buffer, window_width, window_height, fps, current_score, 
                                                   current_level, sound_info, sound_direction, collision_flash, level_up_flash)
            
            current_sound_direction_val = sound_direction 
            
            # --- MODIFIKASI KECEPATAN BOLA ---
            ball_speed_vertical = 6  # Sebelumnya 4, dinaikkan menjadi 6 (atau sesuai keinginanmu)
            ball_speed_horizontal = 3 # Sebelumnya 2, dinaikkan menjadi 3 (atau sesuai keinginanmu)

            if current_sound_direction_val != "neutral":
                center_x += ball_speed_horizontal
                if current_sound_direction_val == "down": 
                    center_y += ball_speed_vertical
                elif current_sound_direction_val == "up": 
                    center_y -= ball_speed_vertical
            
            center_y = max(0, min(center_y, actual_height - resized_ball.shape[0]))
            
            add_ball_trail(center_x + resized_ball.shape[1]//2, center_y + resized_ball.shape[0]//2)
            
            if center_x + resized_ball.shape[1] >= game_view_width:
                if current_level < max_level:
                    current_level += 1
                    current_score += score_per_level
                    level_up_flash = 60
                    print(f"Level Up! Current Level: {current_level}, Score: {current_score}")
                else:
                    print("Max Level Reached!")
                center_x = 0
                center_y = actual_height // 2 - resized_ball.shape[0] // 2

            center_x = max(0, min(center_x, game_view_width - resized_ball.shape[1])) 
            
            top_barrier_factor, bottom_barrier_factor = level_barrier_settings[current_level]
            top_barrier_y = int(actual_height * top_barrier_factor)
            bottom_barrier_y = int(actual_height * bottom_barrier_factor)
            barrier_thickness = 3
            
            # --- MODIFIKASI LOGIKA GAME OVER ---
            if collision_cooldown <= 0:
                if check_collision_with_barriers(center_x, center_y, resized_ball.shape[1], 
                                                 resized_ball.shape[0], top_barrier_y, bottom_barrier_y, barrier_thickness):
                    print(f"Collision! Game Over. Final Score: {current_score}")
                    collision_flash = 30 # Untuk efek visual sesaat sebelum game over
                    game_over = True # Langsung game over
                    # current_score -= barrier_penalty # Penalti skor tidak lagi relevan jika langsung game over
                    # if current_score <= 0: # Cek skor <= 0 tidak lagi relevan
                    # game_over = True
            else:
                collision_cooldown -= 1
            
            if collision_flash > 0:
                collision_flash -= 1
            if level_up_flash > 0:
                level_up_flash -= 1
            
            update_particles()
            
            draw_ball_trail(frame_buffer)
            
            # Pastikan frame_buffer dan resized_ball adalah RGBA jika overlay_transparent mengharapkannya
            # Jika frame_buffer adalah BGR (umumnya dari webcam), dan resized_ball adalah RGBA,
            # overlay_transparent harus menangani ini.
            # Dari kode overlay_transparent, ia menangani background BGR dan overlay RGBA.
            frame_buffer = overlay_transparent(frame_buffer, resized_ball, center_x, center_y)
            
            draw_modern_barriers(frame_buffer, game_view_width, window_height, top_barrier_y, bottom_barrier_y, barrier_thickness)
            
            draw_particles(frame_buffer)
            
            draw_game_view_overlay(frame_buffer, game_view_width, window_height, current_level)
            
            cv2.imshow('Suaraku Terbang', frame_buffer)
            
            key = cv2.waitKey(5) & 0xFF 
            if key == 27 or cv2.getWindowProperty('Suaraku Terbang', cv2.WND_PROP_VISIBLE) < 1:
                break
        
        if game_over: # Hanya tampilkan layar game over jika game_over adalah True
            print(f"Final Score on Game Over Screen: {current_score}") # Skor yang ditampilkan adalah skor saat game over
            game_over_img = draw_modern_game_over(actual_width, actual_height, current_score, max_score_achieved)
            cv2.imshow('Suaraku Terbang', game_over_img)
            cv2.waitKey(0) # Tunggu input sebelum menutup
                
    cap.release()
    cv2.destroyAllWindows()
    sd.stop()

if __name__ == "__main__":
    main()
import cv2
import os
import time
import numpy as np
import mediapipe as mp

# === CONFIG ===
DATASET_DIR = "dataset"
IMG_SIZE = 224
FPS = 30
CAM_INDEX = 0
SAVE_INTERVAL = 1 / FPS

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

# Create dataset folders (A–Z)
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", 
          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
for label in labels:
    os.makedirs(os.path.join(DATASET_DIR, label), exist_ok=True)

# Image capture
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FPS, FPS)
print("Press a letter key (A–Z) to start labeling. Press '.' (period) to quit.")

current_label = None
last_save = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Camera frame not available.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Blank canvas for landmarks only
    canvas = np.zeros_like(frame)

    # Process hand landmarks
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Draw landmarks ONLY (no webcam image)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                canvas,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
            )

    # Display label
    label_text = f"Label: {current_label}" if current_label else "No label selected"
    cv2.putText(canvas, label_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show only the landmark canvas
    cv2.imshow("ASL Dataset Capture (Landmarks Only)", canvas)
    key = cv2.waitKey(1) & 0xFF

    # Keyboard controls
    if key == ord('.'):
        break
    elif ord('a') <= key <= ord('z'):
        current_label = chr(key).upper()

    # Save cropped WIREFRAME ONLY image
    if current_label and results.multi_hand_landmarks and (time.time() - last_save > SAVE_INTERVAL):
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get bounding box
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
        ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)

        xmin, ymin = max(0, xmin - 20), max(0, ymin - 20)
        xmax, ymax = min(w, xmax + 20), min(h, ymax + 20)

        # CROP FROM CANVAS, NOT ORIGINAL FRAME
        wire_crop = canvas[ymin:ymax, xmin:xmax]

        if wire_crop.size > 0:
            wire_crop = cv2.resize(wire_crop, (IMG_SIZE, IMG_SIZE))
            save_path = os.path.join(
                DATASET_DIR, current_label, f"{int(time.time()*1000)}.jpg"
            )
            cv2.imwrite(save_path, wire_crop)
            last_save = time.time()

cap.release()
cv2.destroyAllWindows()
print("Dataset capture complete!")

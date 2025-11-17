import cv2
import os
import time
import mediapipe as mp

# === CONFIG ===
DATASET_DIR = "dataset"            # Root dataset directory
IMG_SIZE = 224                     # Crop size (resize to 224x224)
FPS = 30                           # Target FPS
CAM_INDEX = 0                      # Default webcam index
SAVE_INTERVAL = 1 / FPS            # Time between saves (sec)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

# Create dataset folders (A–Z + nothing)
labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
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
        print("⚠️ Camera frame not available.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Process with MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display current label
    label_text = f"Label: {current_label}" if current_label else "No label selected"
    cv2.putText(frame, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Dataset Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    # Keyboard controls
    if key == ord('.'): # exit key is period
        break
    elif ord('a') <= key <= ord('z'):
        current_label = chr(key).upper()

    # Save cropped hand if label selected and hand detected
    if current_label and results.multi_hand_landmarks and (time.time() - last_save > SAVE_INTERVAL):
        hand_landmarks = results.multi_hand_landmarks[0]
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
        ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)
        xmin, ymin = max(0, xmin - 20), max(0, ymin - 20)
        xmax, ymax = min(w, xmax + 20), min(h, ymax + 20)
        hand_crop = frame[ymin:ymax, xmin:xmax]
        if hand_crop.size > 0:
            hand_crop = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
            save_path = os.path.join(DATASET_DIR, current_label,
                                     f"{int(time.time() * 1000)}.jpg")
            cv2.imwrite(save_path, hand_crop)
            last_save = time.time()

cap.release()
cv2.destroyAllWindows()
print("✅ Dataset capture complete!")

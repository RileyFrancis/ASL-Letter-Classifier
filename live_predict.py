import cv2
import torch
import numpy as np
import torch.nn as nn
import mediapipe as mp
import torch.nn.functional as F
from torchvision import transforms
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/asl_model_dataset.pth"
IMG_SIZE = 256
DRAW_LANDMARKS = True # For best results, leave this as True
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
               'S', 'T', 'U', 'V', 'W', 'X', 'Y'] \
              if "small" not in MODEL_PATH else ['A', 'B']


# GPU MediaPipe Tasks HandLandmarker
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task",
    delegate=python.BaseOptions.Delegate.GPU
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

hand_detector = vision.HandLandmarker.create_from_options(options)

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=len(CLASS_NAMES)):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.serialization.add_safe_globals([CNNClassifier])
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.eval().to(device)
print("Model loaded successfully!")


# Normalization
norm = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Mediapipe Hands wireframe
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

def draw_hand_landmarks(frame, lm_list, width, height):
    # Draw connections
    for start, end in HAND_CONNECTIONS:
        x1 = int(lm_list[start].x * width)
        y1 = int(lm_list[start].y * height)
        x2 = int(lm_list[end].x * width)
        y2 = int(lm_list[end].y * height)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    # Draw points
    for lm in lm_list:
        x = int(lm.x * width)
        y = int(lm.y * height)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


# Capture loop
cap = cv2.VideoCapture(0)
frame_count = 0
print("Starting live ASL recognition — press Q to quit")

SMALL_W, SMALL_H = 320, 240

# main loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    full_h, full_w, _ = frame.shape

    # Downscale for GPU Mediapipe
    small = cv2.resize(frame, (SMALL_W, SMALL_H))
    small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=small_rgb
    )
    results = hand_detector.detect(mp_image)

    prediction_text = "No hand detected"

    # HAND DETECTED
    if results.hand_landmarks:
        lms = results.hand_landmarks[0] # list of all 21 hand landmarks

        # Draw wireframe
        if DRAW_LANDMARKS:
            draw_hand_landmarks(frame, lms, full_w, full_h)

        # Bounding box from tasks landmarks
        xs = [lm.x for lm in lms]
        ys = [lm.y for lm in lms]

        xmin = int(min(xs) * full_w)
        xmax = int(max(xs) * full_w)
        ymin = int(min(ys) * full_h)
        ymax = int(max(ys) * full_h)

        # padding
        pad = 20
        xmin = max(0, xmin - pad)
        ymin = max(0, ymin - pad)
        xmax = min(full_w, xmax + pad)
        ymax = min(full_h, ymax + pad)

        cropped = frame[ymin:ymax, xmin:xmax]

        if cropped.size > 0:
            # Preprocess image
            img = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).float().div(255)
            img = img.permute(2, 0, 1)
            img = norm(img)
            img = img.unsqueeze(0).to(device)

            # CNN inference
            with torch.no_grad():
                outputs = model(img)
                probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

            top_idx = np.argmax(probs)
            prediction_text = f"{CLASS_NAMES[top_idx]} ({probs[top_idx]*100:.1f}%)"

            # class printout
            frame_count += 1
            print("\033c", end="")
            print("Confidence per class:\n")

            for cls, p in zip(CLASS_NAMES, probs):
                bar_len = int(p * 40)
                bar = "█" * bar_len
                print(f"{cls:10s} | {bar:<40s} {p*100:6.2f}%")


    # Display result
    cv2.putText(frame, prediction_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("ASL Live Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

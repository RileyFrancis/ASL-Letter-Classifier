import cv2
import torch
import numpy as np
import torch.nn as nn
import mediapipe as mp
import torch.nn.functional as F
from torchvision import transforms

# === CONFIG ===
MODEL_PATH = "asl_model.pth"   # your saved model
IMG_SIZE = 256                 # must match training
CLASS_NAMES = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ["nothing"]

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)   # → 128x128
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)   # → 64x64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)   # → 32x32

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 32 * 32, 256)  # 131072 inputs
        self.fc2 = nn.Linear(256, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# === DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# === LOAD MODEL ===
model = torch.load(MODEL_PATH, map_location=device)
model.eval().to(device)
print("✅ Model loaded successfully")

# === MEDIAPIPE HANDS ===
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# === PREPROCESSING TRANSFORM ===
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === CAMERA LOOP ===
cap = cv2.VideoCapture(0)
print("🎥 Starting live ASL recognition — press Q to quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("⚠️ Unable to access camera")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction_text = "No hand detected"

    if results.multi_hand_landmarks:
        # Draw hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Crop hand region
        hand_landmarks = results.multi_hand_landmarks[0]
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
        ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)

        # Pad slightly
        pad = 40
        xmin, ymin = max(0, xmin - pad), max(0, ymin - pad)
        xmax, ymax = min(w, xmax + pad), min(h, ymax + pad)

        cropped = frame[ymin:ymax, xmin:xmax]
        if cropped.size > 0:
            # Preprocess
            img_tensor = preprocess(cropped).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                top_idx = np.argmax(probs)
                top_class = CLASS_NAMES[top_idx]
                confidence = probs[top_idx]

            # Format confidences
            confidence_text = f"{top_class} ({confidence*100:.1f}%)"
            prediction_text = confidence_text

            # Display confidence bar chart on console
            print("\033c", end="")  # clear console
            print("📊 Confidence per class:")
            sorted_indices = np.argsort(probs)[::-1]
            for i in sorted_indices[:5]:
                print(f"  {CLASS_NAMES[i]:10s}: {probs[i]*100:6.2f}%")

    # Display frame with prediction
    cv2.putText(frame, prediction_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("ASL Live Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

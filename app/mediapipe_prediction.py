import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from pathlib import Path

# ===========================
# Load CNN Model
# ===========================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "mnist_cnn_model.keras"

print("Loading model from:", MODEL_PATH)
print("Model exists:", MODEL_PATH.exists())

model = tf.keras.models.load_model(MODEL_PATH)
# ==============================
# MediaPipe Hands
# ==============================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ==============================
# Webcam
# ==============================

cap = cv2.VideoCapture(0)

canvas = None

previous_point = None

predicted_digit = ""

# ==============================
# Main Loop
# ==============================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w), dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            cv2.circle(frame, (x, y), 10, (0,255,0), -1)

            if previous_point is None:
                previous_point = (x, y)

            cv2.line(
                canvas,
                previous_point,
                (x, y),
                255,
                15
            )

            previous_point = (x, y)

    else:
        previous_point = None

    # Display canvas

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    merged = cv2.addWeighted(frame, 0.8, canvas_bgr, 0.8, 0)

    cv2.putText(
        merged,
        f"Prediction : {predicted_digit}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

    cv2.imshow("Handwritten Digit Recognition", merged)

    key = cv2.waitKey(1)

    # ==========================
    # Predict Digit
    # Press P
    # ==========================

    if key == ord('p'):

        image = cv2.resize(canvas, (28,28))

        image = image.astype("float32") / 255.0

        image = image.reshape(1,28,28,1)

        prediction = model.predict(image, verbose=0)

        predicted_digit = np.argmax(prediction)

        print("Prediction :", predicted_digit)

    # ==========================
    # Clear Canvas
    # Press C
    # ==========================

    elif key == ord('c'):

        canvas = np.zeros((h,w), dtype=np.uint8)

        predicted_digit = ""

    # ==========================
    # Quit
    # Press Q
    # ==========================

    elif key == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
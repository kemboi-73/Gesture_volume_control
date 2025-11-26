import cv2
import mediapipe as mp
import pyautogui

# Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# Loop through frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            # Get y position of index finger tip
            index_tip_y = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

            # Jump if hand is raised
            if index_tip_y < 0.3:  # Adjust threshold based on webcam distance
                pyautogui.press('space')

            # Duck if hand is low
            elif index_tip_y > 0.7:
                pyautogui.keyDown('down')
            else:
                pyautogui.keyUp('down')

    # Display webcam feed
    cv2.imshow("Dino Hand Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

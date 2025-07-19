import cv2
import mediapipe  as mp
import pyautogui
import threading

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

TIP_IDS = [4, 8, 12, 16, 20]

frame = None
frame_lock = threading.Lock()

def read_frames():
    global frame
    cap = cv2.VideoCapture(0)
    while True:
        success, f = cap.read()
        if success:
            with frame_lock:
                frame = f.copy()

# Start frame capture thread
threading.Thread(target=read_frames, daemon=True).start()

def get_finger_status(hand_landmarks):
    fingers = []

    if hand_landmarks is not None:
        if hand_landmarks.landmark[TIP_IDS[0]].x < hand_landmarks.landmark[TIP_IDS[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if hand_landmarks.landmark[TIP_IDS[id]].y < hand_landmarks.landmark[TIP_IDS[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
    return fingers




# origin at top-leftmost point on the screen with reverse y 

def input_control(input_x, input_y, fingers):
    if input_x is not None and input_y is not None:
        screen_width, screen_height=pyautogui.size()
        current_x=input_x*screen_width
        current_y=input_y*screen_height
        pyautogui.moveTo(current_x, current_y)
        if ( sum(fingers)==2):
            pyautogui.leftClick(current_x, current_y)
        elif( sum(fingers)==3):
            pyautogui.doubleClick(current_x, current_y)


 
# def input_control(x, y):
#     if x is not None and y is not None:
#         screen_w, screen_h = pyautogui.size()
#         px = int(x * screen_w)
#         py = int(y * screen_h)
#         pyautogui.moveTo(px, py)

# Main loop
while True:
    with frame_lock:
        current_frame = frame.copy() if frame is not None else None

    if current_frame is None:
        continue

    img_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    gesture = "No hand"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(current_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers = get_finger_status(hand_landmarks)
            total_fingers = sum(fingers)

            if total_fingers == 5:
                gesture = "Open Palm"
            elif total_fingers == 0:
                gesture = "Fist"
            elif fingers == [1, 0, 0, 0, 0]:
                gesture = "Thumbs Up"
            else:
                gesture = f"{total_fingers} Fingers"

            # Move cursor with index finger
            input_control(hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y, fingers)

    # Show result
    cv2.putText(current_frame, gesture, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow("Hand Gesture", current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




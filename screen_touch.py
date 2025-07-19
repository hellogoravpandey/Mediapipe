import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Finger tip landmarks (thumb to pinky)
TIP_IDS = [4, 8, 12, 16, 20]

def get_finger_status(hand_landmarks):
    fingers = []

    # Thumb: Compare x-coordinates
    if hand_landmarks.landmark[TIP_IDS[0]].x < hand_landmarks.landmark[TIP_IDS[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other four fingers: Compare y-coordinates
    for id in range(1, 5):
        input_control(hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y)
        if hand_landmarks.landmark[TIP_IDS[id]].y < hand_landmarks.landmark[TIP_IDS[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def action(total_fingers):
    if ( total_fingers==5):
        f=open("ACTION.txt","a")
        result="Successful"
        f.write(result)
        f.close()
    
# input controller
# pixel ===>    In 1920x1080 ==> (0, 0)  from x [0, 1919] , y [0, 1079]

def input_control(input_x, input_y):
    pyautogui.move(input_x + 10, input_y + 10)

    while True:
        success, img = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        print("RESULTS\n",results)

        gesture = "No hand"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                finger_status = get_finger_status(hand_landmarks)
                total_fingers = sum(finger_status)

                # Basic gesture interpretation
                if total_fingers == 5:
                    gesture = "Open Palm"
                elif total_fingers == 0:
                    gesture = "Fist"
                elif finger_status == [1, 0, 0, 0, 0]:
                    gesture = "Thumbs Up"
                else:
                    gesture = f"{total_fingers} Fingers"
                action(total_fingers)
        
        # Show gesture name on screen
        cv2.putText(img, gesture, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow("Image Gesture Checker", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

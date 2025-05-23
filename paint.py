import cv2
import mediapipe as mp
import numpy as np
import time
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)

prev_x, prev_y = 0, 0
prev_x2, prev_y2 = 0, 0  # For middle finger in eraser mode
draw_color = (0, 255, 0)
eraser_mode = False
last_color_change = 0

def get_finger_tips(lm_list):
    return {
        'thumb': lm_list[4],
        'index': lm_list[8],
        'middle': lm_list[12],
        'ring': lm_list[16],
        'pinky': lm_list[20]
    }

def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def fingers_up(lm_list):
    return [
        lm_list[8][1] < lm_list[6][1],   # Index
        lm_list[12][1] < lm_list[10][1], # Middle
        lm_list[16][1] < lm_list[14][1], # Ring
        lm_list[20][1] < lm_list[18][1]  # Pinky
    ]

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        handLms = result.multi_hand_landmarks[0]
        lm_list = []

        for id, lm in enumerate(handLms.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append((cx, cy))

        tips = get_finger_tips(lm_list)

        index_finger_up, middle_finger_up, *_ = fingers_up(lm_list)

        # Set eraser mode when both index and middle fingers are up
        eraser_mode = index_finger_up and middle_finger_up

        if eraser_mode:
            # Erase on both index and middle finger tips
            cx_index, cy_index = tips['index']
            cx_middle, cy_middle = tips['middle']

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = cx_index, cy_index
                prev_x2, prev_y2 = cx_middle, cy_middle

            # Erase line following index finger
            cv2.line(canvas, (prev_x, prev_y), (cx_index, cy_index), (0, 0, 0), 30)
            # Erase line following middle finger
            cv2.line(canvas, (prev_x2, prev_y2), (cx_middle, cy_middle), (0, 0, 0), 30)

            prev_x, prev_y = cx_index, cy_index
            prev_x2, prev_y2 = cx_middle, cy_middle

        elif index_finger_up and not middle_finger_up:
            # Draw mode (only index finger up)
            cx, cy = tips['index']
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = cx, cy

            cv2.line(canvas, (prev_x, prev_y), (cx, cy), draw_color, 5)
            prev_x, prev_y = cx, cy

            # Reset middle finger prev points to avoid drawing weird lines when switching modes
            prev_x2, prev_y2 = 0, 0

        else:
            # No drawing or erasing
            prev_x, prev_y = 0, 0
            prev_x2, prev_y2 = 0, 0

        # Gesture: Thumb and Index finger close — change color
        if time.time() - last_color_change > 1.0:
            dist = distance(tips['thumb'], tips['index'])
            if dist < 40:
                draw_color = tuple(np.random.randint(0, 255, 3).tolist())
                last_color_change = time.time()

        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Combine drawing canvas with webcam feed
    frame = cv2.addWeighted(frame, 1, canvas, 0.5, 0)

    cv2.putText(frame, f"Mode: {'Eraser' if eraser_mode else 'Draw'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color if not eraser_mode else (0, 0, 0), 2)

    cv2.imshow("Virtual Paint", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to quit
        break
    elif key == ord('c'):  # Clear canvas
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    elif key == ord('s'):  # Save canvas
        if not os.path.exists("paintings"):
            os.makedirs("paintings")
        filename = f"paintings/drawing_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print(f"Saved: {filename}")

cap.release()
cv2.destroyAllWindows()

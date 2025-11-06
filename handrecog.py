# imports
import cv2
import mediapipe as mp
import time

# video capturing
vid = cv2.VideoCapture(0)
vid.set(3, 1280)

# mediapipe paths
mphands = mp.solutions.hands

# library 'mediapipe/mphands' pipelining 
Hands = mphands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)

# library 'mediapipe/mpdraw' pipelining 
mpdraw = mp.solutions.drawing_utils

# lists
tipIds = [4, 8, 12, 16, 20]

# variables
pTime = 0

# functions
def set_up():
    global pTime  # Need to declare pTime as global to modify it
    
    ret, frame = vid.read()
    
    # convert from BGR to RGB
    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = Hands.process(RGBframe)

    # Create a copy of frame for finger counting display
    img = frame.copy()

    if result.multi_hand_landmarks:
        for handLm in result.multi_hand_landmarks:
            mpdraw.draw_landmarks(frame, handLm, mphands.HAND_CONNECTIONS,
                                  mpdraw.DrawingSpec(color=(0, 0, 255), circle_radius=7,
                                                     thickness=cv2.FILLED),
                                  mpdraw.DrawingSpec(color=(0, 255, 0), thickness=7)
                                  )
            
            # Get landmark positions
            lmList = []
            for id, lm in enumerate(handLm.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if len(lmList) != 0:
                fingers = []

                # thumb tracking (right hand assumed)
                if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # 4 Fingers
                for id in range(1, 5):
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                totalFingers = fingers.count(1)
                print(totalFingers)

                cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                            10, (255, 0, 0), 25)

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Hand Recognition", frame)

    if cv2.waitKey(1) == 27:  # ESC to quit
        cleanup()

# exit function
def cleanup():
    vid.release()
    cv2.destroyAllWindows()
    exit()

# main loop
try:
    while True:
        set_up()
except KeyboardInterrupt:
    cleanup()
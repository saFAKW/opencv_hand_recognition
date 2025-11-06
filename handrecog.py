# imports
import cv2
import mediapipe as mp

# video capturing
vid = cv2.VideoCapture(0) # access to internal webcam, 1,2,3 for external webcams
vid.set(3,1280) #sets the range of the window size

# mediapipe paths
mphands = mp.solutions.hands

# library 'mediapipe/mphands' pipelining 
Hands = mphands.Hands()
Hands = mphands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)

# library 'mediapipe/mpdraw' pipelining 
mpdraw = mp.solutions.drawing_utils

#lists
finger_tips = []
zero = [4,8,12,16,20]
one = [4,12,16,20]
two = [4,16,20]
three = [4,20]

# functions
def set_up():
    ret, frame = vid.read()
    # convert from BGR to RGB
    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # mediapipe and cv work on different colour spectrums
    result = Hands.process(RGBframe)

    if result.multi_hand_landmarks: # branch for hand detection
        for handLm in result.multi_hand_landmarks: # handLm is each hand seen by camera, max. 2
            mpdraw.draw_landmarks(frame, handLm, mphands.HAND_CONNECTIONS,
                                  mpdraw.DrawingSpec(color=(0, 0, 255), circle_radius=7, #blue
                                                     thickness=cv2.FILLED),
                                  mpdraw.DrawingSpec(color=(0, 255, 0), thickness=7) #green lines
                                  )
            for id, lm in enumerate(handLm.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)

                #check for 1 on finger
                if id == 12 or id == 16 or id == 20 or id == 4 or id == 8:
                    barrier = (handLm.landmark[9].y)*h
                    if cy > barrier: # finger down check
                        cv2.circle(frame, (cx, cy), 6, (0, 225, 0), cv2.FILLED) #green
                        finger_tips.append(id)
                        print(finger_tips)
                        if set(zero).issubset(set(finger_tips)):
                            print("0")
                            exit()
                        elif set(one).issubset(set(finger_tips)):
                            print("1")
                        elif set(two).issubset(set(finger_tips)):
                            print("2")
                        elif set(three).issubset(set(finger_tips)):
                            print("3")
                        elif 4 in finger_tips:
                            print("4")
                        else:
                            print("5")
                finger_tips.clear()
                    # cv2.line(frame, (cx, cy), (Tx, Ty), (255, 0, 0), 5 )

    cv2.imshow("Hand Recognition", frame) # intialises window and default framerate

    if cv2.waitKey(1) == 27:  # ESC to quit
        exit()

#check no. of fingers function
def check_fingers():
    pass

#exit function
def exit():
    vid.release()
    cv2.destroyAllWindows() # closes program

# main loop
while True:
    set_up()


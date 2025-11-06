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
monitor = {
    12: 0,
    16: 0,
    20: 0
}

# main loop
while True:
    ret, frame = vid.read()
    # convert from BGR to RGB
    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # mediapipe and cv work on different colour spectrums
    result = Hands.process(RGBframe)

    if result.multi_hand_landmarks: # branch for hand detection
        for handLm in result.multi_hand_landmarks: # handLm is each hand seen by camera, max. 2
            mpdraw.draw_landmarks(frame, handLm, mphands.HAND_CONNECTIONS,
                                  mpdraw.DrawingSpec(color=(0, 0, 255), circle_radius=7,
                                                     thickness=cv2.FILLED),
                                  mpdraw.DrawingSpec(color=(0, 255, 0), thickness=7)
                                  )
            for id, lm in enumerate(handLm.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                count = 0
                #check for 1 on finger
                if id == 12 or id == 16 or id == 20 or id == 4:
                    barrier = (handLm.landmark[9].y)*h

                    if cy < barrier: # finger down check
                        cv2.circle(frame, (cx, cy), 6, (0, 225, 0), cv2.FILLED) 
                        monitor[id] = cy # only if less than 9 then the spot is filled
                        count += 1
                        print(count)

                    # cv2.line(frame, (cx, cy), (Tx, Ty), (255, 0, 0), 5 )

    cv2.imshow("Hand Recognition", frame) # intialises window and default framerate


    if cv2.waitKey(1) == 27:  # ESC to quit
        vid.release()
        cv2.destroyAllWindows() # closes program 

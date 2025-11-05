#imports
import cv2
import mediapipe as mp

#video capturing
vid = cv2.VideoCapture(0) # access to internal webcam, 1,2,3 for external webcams
vid.set(3,960) #sets the range of the window size

#mediapipe paths
mphands = mp.solutions.hands

#library 'mphands' pipelining 
Hands = mphands.Hands()
Hands = mphands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)

#main loop
while True:
    ret, frame = vid.read()
    #convert from BGR to RGB
    RGBframe = cv2.cvtColor(frame, cv2.COLOUR_BGR2RGB) #mediapipe and cv work on different colour spectrums


    cv2.imshow("Hand Recognition", frame)


    if cv2.waitKey(1) == 27:  # ESC to quit
        vid.release()
        cv2.destroyAllWindows() # closes program 

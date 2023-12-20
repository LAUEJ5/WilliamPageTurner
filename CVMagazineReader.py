import tkinter as tk
from tkinter import Canvas, Label, Toplevel
from PIL import Image, ImageTk
import threading
import time
import cv2
import mediapipe as mp
import time
import math
import numpy as np
import fitz 

##  GLOBALS  ##
current_page = 0
pastInTime = 0
allPoints = []
points = np.zeros((0, 5))
pTime = 0
cTime = 0
lastPageChange = 0


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, 1, self.detectionConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    # function to draw hands in image
    def findHands(self, img, draw=True):
        if img is None:
            return img
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    # gets specific hand points
    def getHandLabels(self, img):
        handsType = []
        numHands = 0
        if self.results.multi_handedness:
            numHands = len(self.results.multi_handedness)
            for hand in self.results.multi_handedness:
                handsType.append(hand.classification[0].label)
 
        return handsType, numHands

    # function to return a list of points for both left hand and right
    def findPosition(self, img, handsType, numHands, draw=True):
        lmListLeft = []
        lmListRight = []

        # if no hands are found
        if (numHands == 0):
            return lmListLeft, lmListRight

        if self.results.multi_hand_landmarks:
            for hand, handType in zip(self.results.multi_hand_landmarks, handsType):

                # if right hand, draw black points
                if (handType == "Right"):
                    for id, lm in enumerate(hand.landmark):

                        h, w, c = img.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        lmListRight.append([id, cx, cy])
                        if draw:
                            cv2.circle(img, (cx, cy), 7,
                                       (0, 0, 0), cv2.FILLED)

                # if left hand, draw red points
                if (handType == "Left"):
                    for id, lm in enumerate(hand.landmark):

                        h, w, c = img.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        lmListLeft.append([id, cx, cy])
                        if draw:
                            cv2.circle(img, (cx, cy), 7,
                                       (0, 255, 0), cv2.FILLED)

        # return updated lists of hand points
        return lmListLeft, lmListRight
    

def getHandTotal(lmList, cTime, scalar):
    global pastInTime
    total = 0

    # Thumb distace
    valx = (lmList[6][1] - lmList[4][1])/scalar
    valy = (lmList[6][2] - lmList[4][2])/scalar
    thumb = int(100 * math.sqrt((valy**2) + (valx**2)))

    # Pointer distance
    valx = (lmList[5][1] - lmList[8][1])/scalar
    valy = (lmList[5][2] - lmList[8][2])/scalar
    pointer = int(100 * math.sqrt((valy**2) + (valx**2)))

    # Middle Finger
    valx = (lmList[9][1] - lmList[12][1])/scalar
    valy = (lmList[9][2] - lmList[12][2])/scalar
    middle = int(100 * math.sqrt((valy**2) + (valx**2)))

    # Ring Finger
    valx = (lmList[16][1] - lmList[13][1])/scalar
    valy = (lmList[16][2] - lmList[13][2])/scalar
    ring = int(100 * math.sqrt((valy**2) + (valx**2)))

    # Pinky
    valx = (lmList[20][1] - lmList[17][1])/scalar
    valy = (lmList[20][2] - lmList[17][2])/scalar
    pinky = int(100 * math.sqrt((valy**2) + (valx**2)))

    total = thumb + pointer + middle + ring + pinky
    return total


# Funcion to read both hands for driving inputs
def readBothHands(lmListLeft, lmListRight, img, cTime):
    global currentPage
    global pastInTime
    global points
    global allPoints
    global lastPageChange

    lastPageChange = lastPageChange + 1
    # Get Scalar based on right hand
    valy = lmListRight[5][2] - lmListRight[17][2]
    valx = lmListRight[5][1] - lmListRight[17][1]
    rightScaler = math.sqrt((valy**2) + (valx**2))

    # Get Scalar based on left hand
    valy = lmListLeft[5][2] - lmListLeft[17][2]
    valx = lmListLeft[5][1] - lmListLeft[17][1]
    leftScaler = math.sqrt((valy**2) + (valx**2))

    if rightScaler == 0 or leftScaler == 0:
        return

    totalLeft = getHandTotal(lmList=lmListLeft, cTime=cTime, scalar=leftScaler)
    totalRight = getHandTotal(lmList=lmListRight, cTime=cTime, scalar=rightScaler)
    if(totalLeft < 200):
        if cTime - pastInTime > 0:
            print(lastPageChange)
            if lastPageChange > 20:
                lastPageChange = 0
                print("Turn page LEFT")
                go_to_previous_page()     
                
                
                
    if(totalRight < 200):
        if cTime - pastInTime > 0:
            print(lastPageChange)
            if lastPageChange > 20:
                lastPageChange = 0
                print("Turn page RIGHT")
                go_to_next_page()
                
            

# Function to get the PDF page as an image
def get_pdf_page(page_number):
    page = pdf_document.load_page(page_number)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

# Function to update the displayed PDF page
def update_pdf_display(page_number):
    img = get_pdf_page(page_number)
    photo = ImageTk.PhotoImage(image=img)
    canvas.image = photo  # Keep a reference so it's not garbage collected
    canvas.create_image(10, 10, image=photo, anchor='nw')

# Function for the counter
def counter():
    count = 0
    global pTime
    global cTime
    while True:
        time.sleep(1.0 / 60)

        # **** Hand Detection **** #
        success, img = cap.read()
        if img is None:
            continue
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        handsType, numHands = detector.getHandLabels(img)
        lmListLeft, lmListRight = detector.findPosition(
            img, handsType, numHands)

        # Reading both hands
        if len(lmListLeft) != 0 and len(lmListRight) != 0:
            readBothHands(lmListLeft, lmListRight, img, cTime)

        # No hands
        else:
            # print("no hands")
            for i in range(len(allPoints)):
                for f in range(len(allPoints[i])):
                    if f != 0:
                        cv2.line(img, (int(allPoints[i][f][0]), int(allPoints[i][f][1])), (int(allPoints[i][f - 1][0]), int(
                            allPoints[i][f - 1][1])), (int(allPoints[i][f][2]), int(allPoints[i][f][3]), int(allPoints[i][f][4])), 3, cv2.FILLED)

        # show FPS in top left
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
        # **** Hand Detection **** #

# Function to handle page changes
def go_to_next_page():
    global current_page
    print(current_page, ", ", len(pdf_document))
    if current_page < len(pdf_document) - 1:
        current_page += 1
        update_pdf_display(current_page)
        

def go_to_previous_page():
    global current_page
    if current_page > 0:
        current_page -= 1
        update_pdf_display(current_page)
        

        

def main():
    global canvas
    global pdf_document
    global timer_label
    global current_page
    
    # **** Hand Detection **** #
    global cap
    global detector
    global pTime
    global cTime
    # Prepare webcam and handdetector
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    success, img = cap.read()
    h, w, c = img.shape
    # **** Hand Detection **** #


    # Load your PDF
    pdf_path = 'The-Grand-Budapest-Hotel-.pdf'
    pdf_document = fitz.open(pdf_path)
    # Initialize the tkinter windows
    main_window = tk.Tk()
    main_window.title("PDF Viewer")

    # Start the counter thread
    counter_thread = threading.Thread(target=counter, daemon=True)
    counter_thread.start()

    # Create a canvas to display the image in the main window
    canvas = Canvas(main_window, width=800, height=600)
    canvas.pack()

    # Initialize with the first page
    update_pdf_display(current_page)

    # Buttons to navigate through the PDF
    next_button = tk.Button(main_window, text="Next", command=go_to_next_page)
    next_button.pack(side='right')

    prev_button = tk.Button(main_window, text="Previous", command=go_to_previous_page)
    prev_button.pack(side='left')

    # Start the tkinter loop
    main_window.mainloop()

if __name__ == "__main__":
    main()

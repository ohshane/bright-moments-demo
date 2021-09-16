from __future__ import print_function
import cv2 as cv
import argparse


def detectAndDisplay(frame, noise=False):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    faceROI = None
    
    if noise == 'upper-block':
        for (x,y,w,h) in faces:
            frame = cv.rectangle(frame, (x, y), (x+w, y+int(h/2)), (0,0,0), -1)
            break
    elif noise == 'under-block':
        for (x,y,w,h) in faces:
            frame = cv.rectangle(frame, (x, y+int(h/2)), (x+w, y+h), (0,0,0), -1)
            break
    elif noise == 'eye-block':
        for (x,y,w,h) in faces:
            frame = cv.rectangle(frame, (x, y+int(h/3)), (x+w, y+int(h/2)), (0,0,0), -1)
            break
    elif noise == 'nose-block':
        for (x,y,w,h) in faces:
            frame = cv.rectangle(frame, (x, y+int(h/2)), (x+w, y+int(h*2/3)), (0,0,0), -1)
            break
    elif noise == 'mouth-block':
        for (x,y,w,h) in faces:
            frame = cv.rectangle(frame, (x, y+int(h*2/3)), (x+w, y+h), (0,0,0), -1)
            break
    elif noise == 'upper-gaussian':
        for (x,y,w,h) in faces:
            blurred = cv.GaussianBlur(frame, (25,25), 0)
            frame[y:y+int(h/2),x:x+w] = blurred[y:y+int(h/2),x:x+w]
            break
    elif noise == 'under-gaussian':
        for (x,y,w,h) in faces:
            blurred = cv.GaussianBlur(frame, (25,25), 0)
            frame[y+int(h/2):y+h,x:x+w] = blurred[y+int(h/2):y+h,x:x+w]
            break
    elif noise == 'eye-gaussian':
        for (x,y,w,h) in faces:
            blurred = cv.GaussianBlur(frame, (25,25), 0)
            frame[y+int(h/3):y+int(h/2),x:x+w] = blurred[y+int(h/3):y+int(h/2),x:x+w]
            break
    elif noise == 'nose-gaussian':
        for (x,y,w,h) in faces:
            blurred = cv.GaussianBlur(frame, (25,25), 0)
            frame[y+int(h/2):y+int(h*2/3),x:x+w] = blurred[y+int(h/2):y+int(h*2/3),x:x+w]
            break
    elif noise == 'mouth-gaussian':
        for (x,y,w,h) in faces:
            blurred = cv.GaussianBlur(frame, (25,25), 0)
            frame[y+int(h*2/3):y+h,x:x+w] = blurred[y+int(h*2/3):y+h,x:x+w]
            break

    for (x,y,w,h) in faces:
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)
        # frame = cv.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 1)
        faceROI = frame_gray[y:y+h,x:x+w]
    return frame, faceROI, faces
    
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()

#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
camera_device = args.camera

#-- 2. Read the video stream
if __name__ == "__main__":
    cap = cv.VideoCapture(camera_device)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        detectAndDisplay(frame)
        if cv.waitKey(10) == 27:
            break
import numpy as np
import cv2 as cv
#import os          this is to loop over directories, we are not doing that

haar_cascade = cv.CascadeClassifier('haar_frontalface.xml')   #<path to xml file (haar cascade classifier)> for identifying faces

#load our features and label rate
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']     #to get the mapping
#features = np.load('features.npy')     #if you wanted to use this again you could with np.load. since data types are objects set allow_pickles=True. see below
#features = np.load('features.npy', allow_pickle=True)
#labels = np.load('labels.npy')

#read in face_trained.yml file
face_recognizer = cv.face.LBPHFaceRecognizer_create()   #we will instanciate our face recognizer
face_recognizer.read('face_trained.yml')#give it path to yml source file)

#create img variable
img=cv.imread(r'C:\Users\Hasaan\Documents\OpenCV\validate_faces\nottrain3.jpg')

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

#detect face in image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)    #<gray img>, <scale factor>, minneighbors

#loop over every face in faces_rect to find the region of interest (the face)
for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    #predict using the face recognizer
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {label} with a confidence of {confidence}')
    print(f'Label = {people[label]} with a confidence of {confidence}')

    #put some text on the image so we can see whats really going on
    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), thickness=2)    #<image your putting text on>, <text you want>, <origin of text>, <font ex. hershey complex >, <font scale>, <color>,<thickness>
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2 )    #draw a rectangle over the image

#display the image
cv.imshow('Detected Face',img )


cv.waitKey(0)



#it doesnt work great tbh. not the most acurate. gets stuff wrong 
#bc (1) we only had 100 images to train the recognizer on 
#and (2) because we were not using a deep learning model
#
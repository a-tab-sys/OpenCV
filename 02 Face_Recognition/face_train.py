import os                                   #os module lets us interact with the operating system           
import cv2 as cv
import numpy as np                          #library that supports in dealing with multi-dimensional arrays

# we are going to pass in images of 5 celebrities and train a recognizer on those images

# create a list of all the peple in the image
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
print(people)

# or create list by looping over every folder in the train_faces_photos folder
# the r before a path in Python creates a raw string. Raw strings treat backslashes (\) as literal characters instead of escape characters
# the os.listdir() function returns a list of the names of all files and subdirectories in the given directory (path). It does not include the full paths, just the names
# the output of the program will be: ['image1.jpg', 'image2.png', 'subfolder1', 'subfolder2']
p=[]
for i in os.listdir(r'C:\Users\Hasaan\Documents\OpenCV\train_faces_photos'):      
    p.append(i)
print(p)

# read in the haar xml file by createing a haar cascade variable
haar_cascade = cv.CascadeClassifier('haar_cascade\haar_frontalface.xml')

# create a function that will loop over every folder in the train_faces_photos folder
# and loop over every image in the subsequesnt folders 
# it will grab the face from the images and add it to our training set

# our training set will consists of 2 lists, 
# the first list is called features which is an image array of the faces
# the second list will be our corresponding labels (celebrity names)

DIR=r'C:\Users\Hasaan\Documents\OpenCV\train_faces_photos'

features=[]     # empty lists for now
labels=[]

def create_train():
    for person in people:                             # loop over every person in people list
        path=os.path.join(DIR, person)                # grab the path for each person, each folder inside train_faces folder
        label=people.index(person)
        
        # now we are inside each folder, we will loop over every image in that folder
        for img in os.listdir(path):
            img_path=os.path.join(path, img)    # going to grab the image path by joining path and img

            # now that we have the path to an image, we are going to read in that image fom this path
            img_array=cv.imread(img_path)
            if img_array is None:
                continue 
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)    # convert image to grayscale

            # now we can try to detect the faces in the image
            # detectMultiScale is an instance of CascadeClassifier class
            # it will take in the image and use variables scaleFactor and minNeighbors to detect a face
            # and return the rectangular coordinates of that face as a list to the variable faces_rect
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            # since faces_rect has the rectangular coordinates for the faces that are present in the image
            # we can loop over this list and grab the faces region of inerest and
            # crop out the face in the image
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]       # y to y+h , x to x+w
                # now that we have  face region of interest we can append that to features list, and append the correspoiding label (caleb name) to the label list
                features.append(faces_roi)
                labels.append(label)        # this label variable is the index of the people list
                                            # converting a label to numerical values using mapping between string and numerical label reduces strain on computer
                                            # ex. label for ben afflech woul be 0, elton john images would have label of 1

create_train()     # printing/running function

# now we see how many features(photos) we had in total and how many associated labels
print(f'length of the features = {len(features)}')
print(f'length of the labels= {len(labels)}')
print('---------Training Complete--------')

# NEXT STEP
# now that our features and labels list is appended, we can train our recognizer on it
# cv.face.LBPHFaceRecognizer_create(): this function initializes an object for the LBPH Face Recognizer in OpenCV.
# LBPH is one of the commonly used algorithms for face recognition. It is based on Local Binary Patterns (LBP), a texture descriptor, and creates histograms of patterns around pixel neighborhoods to represent and recognize faces.

# converting features and labels list to numpy arrays
features = np.array(features, dtype='object')
labels = np.array(labels)

# The variable face_recognizer becomes an instance of the LBPH face recognizer class.
#face_recognizer = cv.face.LBPHFaceRecognizer.create() 
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# train the recognizer on the feature and labels list
face_recognizer.train(features, labels)

# if we wanted to use this trained model, we would have to 
# repeat the aove process, but opencv lets you save the trained
# model in another file/directry/elsewhere by using that particular YML source file
face_recognizer.save('face_trained.yml')    #give it a path to a YML source file

# save fetures and labels list
np.save('features.npy', features)
np.save('labels.npy', labels)

# now when this is run you will have 3 additional files in your directory:
# features.npy, labels.npy, face_trained.yml
            
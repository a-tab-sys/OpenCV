import cv2 as cv

# Check OpenCV version
print(cv.__version__)

# Check if the `face` module exists
if hasattr(cv, 'face'):
    print("The 'face' module is available.")
else:
    print("The 'face' module is NOT available.")


# Check if the `LBPHFaceRecognizer_create` module exists
if hasattr(cv.face, 'LBPHFaceRecognizer_create'):
    print("The 'face' module is available.")
else:
    print("The 'LBPHFaceRecognizer_create' module is NOT available.")

# checkwhich version you have of OpenCV
print(cv.__version__)

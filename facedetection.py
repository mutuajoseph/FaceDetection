import cv2

# adding our cascade
faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")


# read an image 
img = cv2.imread("Resources/download.jpeg")

# resize the image
imgResize = cv2.resize(img, (600,400))

# Convert image to grayscale 
imgGray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)


# find the faces in the image 

face = faceCascade.detectMultiScale(imgGray, 1.1, 4)

# create a bounding box across the faces detected
# loop through all the faces in the image 
# define the starting point
# define the corner points  
# define a color
# define the thickness

for (x,y,w,h) in face:
    cv2.rectangle(imgResize, (x,y), (x+w, y+h), (255,0,0), 2)
    cv2.putText(imgResize, "face", (x+30, y-7), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

# show/ display it
cv2.imshow("Faces", imgResize)

# delay
cv2.waitKey(0)
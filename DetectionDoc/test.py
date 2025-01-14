import cv2
import numpy as np
import Run_a_PATH_in_VSCODE
import time

path = r'C:\Workdir\Perso\TRAVAIL\Perso\DetectionDoc\asset\image.png'



# Let's load a simple image with 3 black squares 
image = cv2.imread(path) 
cv2.waitKey(0) 

# Grayscale 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

# Find Canny edges 
edged = cv2.Canny(gray, 30, 200) 
cv2.waitKey(0) 

# Finding Contours
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

cv2.imshow('Canny Edges After Contouring', edged) 
cv2.waitKey(0) 

cv2.destroyAllWindows() 

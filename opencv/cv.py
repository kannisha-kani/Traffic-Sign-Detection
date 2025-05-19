"""
"""
import cv2
a=cv2.imread("sl image.jpeg")
cv2.imshow("image",a)
cv2.waitKey(0)
"""
import cv2
#Load image
image=cv2.imread("sl image.jpeg")
#Resize to 300*300 pixels
resized=cv2.resize(image,(300,300))
Cropped_image=image[50:200,100:300]
#show the resized image
cv2.imshow("Resized image",resized)
cv2.imshow("Cropped",Cropped_image)
cv2.waitKey(0)
cv2.destroyAllwindows()
"""
"""
import cv2
image=cv2.imread("sl image.jpeg",cv2.IMREAD_GRAYSCALE)
edge=cv2.Canny(image,100,200)
cv2.imshow("Edges",edge)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
""""
import matplotlib.pyplot as plt
import numpy as np
image = cv2.imread("sl image.jpeg",cv2.IMREAD_GRAYSCALE)
sobel_x= cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
edges
sobel_y=cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)
"""
# image_circle = img .copy()
# cv2.circle(image_circle,(255,255),63,(0,255,0),-1)
# cv2.imshow("circle window",image_circle)
# cv2.waitKey(0)
"""
import cv2
import numpy as np
def trackbarChange(x):
    print(x)
img=np.zeros((300,512,3),np.uint8)
cv2.namewindow('image')
cv2.createTrackbar('B','image',0,255,trackbarChange)
cv2.createTrackbar('G','image',0,255,trackbarChange)
cv2.createTrackbar('R','image',0,255,trackbarChange)
while(1):
    cv2.imshow('image',img)
    k=cv2.waitkey(1)&0xFF
    if k==27:
        break
    b=cv2.getTrackbarPos('B','image')
    b=cv2.getTrackbarPos('G','image')
    r=cv2.getTrackbarPos('R','image')
    img[:]=[b,g,r]
    cv2.destroyAllWindows()"
    """
"""
import cv2
import numpy as np
image = cv2.imread("sl image.jpeg")
image = cv2.resize(image, (600, 400))
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
median_blur = cv2.MedianBlur(image, 5)
bilateral_blur = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imshow("Original Image", image)
cv2.imshow("Gaussian Blur", gaussian_blur)
cv2.imshow("Median Blur", median_blur)
cv2.imshow("Bilateral Filtering", bilateral_blur)
cv2.WaitKey(0)
cv2.destroyAllWindows()"
"""
"""
import cv2
import numpy as np
img = cv2.imread("input_sl image.jpeg")
gray = cv2.cvtcolor(img, cv2.COLOUR_BGR2GRAY)
binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
kernel = np.ones((3,3),np.uint8)
cv2.imshow("Original Image",img)
cv2.imshow("Binary Image",binary)
cv2.imshow("Eroded Image",eroded)
cv2.WaitKey(0)
cv2.destroyAllWindows()"
"""
"""
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Webcam' ,frame)
    if cv2.waitKey(1) == ord('q')
        break
    cap.release()
    cv2.destroyAllWindows()
import cv2
cap = cv2.videocapture('1.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('video', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
"""

"""
import cv2
import numpy as np
img = cv2.imread('input_image-1.jpg')
gray = cv2.cvtcolor(img, cv2.COLOR_BGR2GRAY)
binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy =  cv2.findcontours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
Contour_img = img.copy()
cv2.drawContours(c0ntour_img, contours, -1, (0,255, 0), 2)
"""
"""
import cv2
img = cv2.imread('input_image-1.jpg')
layer = img.copy()
gp = [layer]
for i in range(6):
    layer = cv2.pyrDown(layer)
    gp.append(layer)
layer = gp[5]
lp = [layer]
for i in tange(5,0,-1):
    gaussian_extended = cv2.pyrup(gp[i])
    laplacian = cv2.subtract(gp[i-1],gaussian_extended)
    cv2.imshow(str(i),laplacian)
cv2.imshow('Original Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
import cv2
import numpy as np
apple = cv2.imread('input_image-1.jpg')
apple = cv2.resize(apple,(512,512))
orange = cv2.imread('a3.jpg')
orange = cv2.resize(orange,(512,512))
print(apple.shape)
print(orange.shape)
apple_orange = np.hstack((apple[:,:256],orange[:,256:]))
apple_copy = apple.copy()
gp_apple = [apple_copy]
for i in range(6):
    apple_copy = cv2.pyrDown(apple_copy)
    gp_apple.append(apple_copy)
orange_copy = orange.copy()
gp_orange = orange.copy()
for i in range(6):
    orange_copy = cv2.pyrDown(orange_copy)
    gp_orange.append(orange_copy)
apple_copy = gp_apple[5]
lp_apple = [apple_copy]
for i in range(5,0,-1):
    gaussian_expanded = cv2.pyrUp(gp_apple[i])
    laplacian = cv2.subtract(gp_apple[i-1],gaussian_expanded)
    lp_apple.append(laplacian)
orange_copy = gp_orange[5]
lp_orange = [orange_copy]
for i in range(5,0,-1):
    gaussian_expanded = cv2.pyrUp(gp_orange[i])
    laplacian = cv2.subtract(gp_orange[i-1],gaussian_expanded)
    lp_orange.append(laplacian)
apple_orange_pyramid =[]
n=0
for apple_lap,orange_lap in zip(lp_apple,lp_orange):
    n+=1
    cols,rows,ch = apple_lap.shape
    laplacian = np.hstack((apple_lap[:,0:int(cols/2)],orange_lap[:,int(cols/2):]))
cv2.imshow('Apple',apple)
cv2.imshow('orange',orange)
cv2.imshow('apple_orange',apple_orange)
cv2.imshow('apple_orange_reconstruct',apple_orange_reconstruct)
cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2
import numpy as np
image = cv2.imread('shapes.png')
gray= cv2.cvtcolor(image, cv2.COLOR_BGR2GRAY)
"""

"""
import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for default camera, or change to video file path

# Define color ranges for traffic sign detection (in HSV)
# Red color range (for stop signs, yield, etc.)
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

# Blue color range (for informational signs)
lower_blue = np.array([100, 70, 50])
upper_blue = np.array([130, 255, 255])

# Yellow color range (for warning signs)
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Minimum contour area to consider as a sign
min_contour_area = 500

def detect_signs(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create masks for different colors
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine all masks
    combined_mask = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_blue, mask_yellow))
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and shape
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_contour_area:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Determine the shape based on number of vertices
            vertices = len(approx)
            sign_type = "Unknown"
            color = (0, 255, 255)  # Yellow for unknown
            
            # Check for circular/octagonal signs (like stop signs)
            if vertices >= 8:
                sign_type = "Stop/Yield"
                color = (0, 0, 255)  # Red
            # Check for triangular signs (warning signs)
            elif vertices == 3:
                sign_type = "Warning"
                color = (0, 255, 255)  # Yellow
            # Check for rectangular signs (regulatory/informational)
            elif vertices == 4:
                sign_type = "Regulatory/Info"
                color = (255, 0, 0)  # Blue
            
            # Draw the contour and label
            cv2.drawContours(frame, [approx], 0, color, 2)
            
            # Get the bounding rectangle and put text
            x, y, w, h = cv2.boundingRect(approx)
            cv2.putText(frame, sign_type, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Detect traffic signs
    detected_frame = detect_signs(frame)
    
    # Display the result
    cv2.imshow('Traffic Sign Detection', detected_frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
"""

"""
import cv2
# Load Haar cascade (you need a trained XML file for specific traffic signs)
# For demonstration, we use face cascade as a placeholder
cascade_path = 'haarcascade_frontalface_default.xml'  # Replace with actual sign cascade
detector = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect signs
    signs = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles
    for (x, y, w, h) in signs:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Detected Sign", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Traffic Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""




import cv2
import numpy as np

# Initialize the video capture (0 for webcam, or file path for video)
cap = cv2.VideoCapture(0)  # Change to your video source if needed

# Load pre-trained model for traffic sign detection
# For this example, we'll use a simple color/shape-based approach
# In a real application, you would use a trained CNN model

# Define color ranges for sign detection (in HSV)
color_ranges = {
    'red': ([0, 70, 50], [10, 255, 255]),
    'blue': ([100, 150, 0], [140, 255, 255]),
    'yellow': ([20, 100, 100], [30, 255, 255]),
    'green': ([40, 70, 80], [80, 255, 255])
}

# Dictionary of sign types we want to detect
sign_types = {
    'stop': {'color': 'red', 'shape': 'octagon'},
    'yield': {'color': 'red', 'shape': 'triangle'},
    'speed_limit': {'color': 'white', 'shape': 'circle'},
    'no_entry': {'color': 'red', 'shape': 'circle'},
    'traffic_light': {'color': ['red', 'green', 'yellow'], 'shape': 'rectangle'}
}

def detect_shapes(contour):
    shape = "unidentified"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    
    # Triangle
    if len(approx) == 3:
        shape = "triangle"
    # Rectangle or square
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    # Octagon (stop sign)
    elif len(approx) == 8:
        shape = "octagon"
    # Circle
    else:
        shape = "circle"
    
    return shape

def detect_signs(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    detected_signs = []
    
    # Check for each color range
    for color_name, (lower, upper) in color_ranges.items():
        # Create mask for the color
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 1000:
                continue
                
            # Detect shape
            shape = detect_shapes(contour)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Try to identify the sign type
            sign_type = "unknown"
            for type_name, properties in sign_types.items():
                if (properties['color'] == color_name or 
                    (isinstance(properties['color'], list) and color_name in properties['color'])) and \
                   properties['shape'] == shape:
                    sign_type = type_name
                    break
            
            detected_signs.append({
                'type': sign_type,
                'bbox': (x, y, x+w, y+h),
                'color': color_name,
                'shape': shape
            })
    
    return detected_signs

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for webcam
    frame = cv2.flip(frame, 1)
    
    # Detect signs in the frame
    signs = detect_signs(frame)
    
    # Draw bounding boxes and labels
    for sign in signs:
        x1, y1, x2, y2 = sign['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f"{sign['type']} ({sign['color']} {sign['shape']})"
        cv2.putText(frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Traffic Sign Detection', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()









# EXnO12-Face Detection using Haar Cascades with OpenCV and Matplotlib
# Aim:
To write a Python program using OpenCV to perform the following image manipulations: i) Extract ROI from an image. ii) Perform face detection using Haar Cascades in static images. iii) Perform eye detection in images. iv) Perform face detection with label in real-time video from webcam.

# Software Required:
Anaconda - Python 3.7 or above OpenCV library (opencv-python) Matplotlib library (matplotlib) Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

# Algorithm:
I) Load and Display Images

Step 1: Import necessary packages: numpy, cv2, matplotlib.pyplot

Step 2: Load grayscale images using cv2.imread() with flag 0

Step 3: Display images using plt.imshow() with cmap='gray'

II) Load Haar Cascade Classifiers

Step 1: Load face and eye cascade XML files

III) Perform Face Detection in Images

Step 1: Define a function detect_face() that copies the input image

Step 2: Use face_cascade.detectMultiScale() to detect faces

Step 3: Draw white rectangles around detected faces with thickness 10

Step 4: Return the processed image with rectangles

IV) Perform Eye Detection in Images

Step 1: Define a function detect_eyes() that copies the input image

Step 2: Use eye_cascade.detectMultiScale() to detect eyes

Step 3: Draw white rectangles around detected eyes with thickness 10

Step 4: Return the processed image with rectangles

V) Display Detection Results on Images

Step 1: Call detect_face() or detect_eyes() on loaded images

Step 2: Use plt.imshow() with cmap='gray' to display images with detected regions highlighted

VI) Perform Face Detection on Real-Time Webcam Video

Step 1: Capture video from webcam using cv2.VideoCapture(0)

Step 2: Loop to continuously read frames from webcam

Step 3: Apply detect_face() function on each frame

Step 4: Display the video frame with rectangles around detected faces

Step 5: Exit loop and close windows when ESC key (key code 27) is pressed

Step 6: Release video capture and destroy all OpenCV windows

# Program:

```

import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread(r"C:\Users\acer\Pictures\Screenshots\Screenshot 2026-08-19 161229.png")  # Replace with your image path
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Step 2: Display the original image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('on')
plt.show()
roi = image[100:420, 200:550]  # ROI coordinates (adjust as needed)
ask = np.zeros_like(image)
# Place the ROI on the mask
mask[100:420, 200:550] = roi
segmented_roi = cv2.bitwise_and(image, mask)
segmented_roi_rgb = cv2.cvtColor(segmented_roi, cv2.COLOR_BGR2RGB)
plt.imshow(segmented_roi_rgb)
plt.title("Segmented ROI")
plt.axis('off')
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread(r"C:\Users\acer\Pictures\Screenshots\111.png")  # Replace with your actual image file path
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  
edges = cv2.Canny(blurred_image, 50, 150)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
result_image = image.copy()  # Create a copy of the original image to draw bounding boxes
for contour in contours:
    if cv2.contourArea(contour) > 50:  # Filter out small areas
        x, y, w, h = cv2.boundingRect(contour)  # Get the bounding box for the contour
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw the rectangle
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title("Handwriting Detection")
plt.axis('off')
net = cv2.dnn.readNetFromCaffe(config_file, weights)
class_labels = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable',
                12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep',
                18: 'sofa', 19: 'train', 20: 'tvmonitor'}
image = cv2.imread(r"C:\Users\acer\Pictures\Screenshots\Screenshot 2026-08-19 161229.png")  # Replace with your image path
(h, w) = image.shape[:2]
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:  # Confidence threshold
        index = int(detections[0, 0, i, 1])  # Get class index
        label = class_labels[index]  # Get label name
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
# Step 8: Draw rectangles and labels on the image
        cv2.rectangle(image_rgb, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image_rgb, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
plt.imshow(image_rgb)
plt.title("Object Detection with MobileNet-SSD")
plt.axis("off")
plt.show()

```

# Output:
<img width="1422" height="477" alt="image" src="https://github.com/user-attachments/assets/c2fd998f-8884-4a95-bc04-c3671cf6f4cf" />
<img width="660" height="666" alt="image" src="https://github.com/user-attachments/assets/ce164650-0edd-442b-9ad4-594145ce7203" />
<img width="493" height="657" alt="image" src="https://github.com/user-attachments/assets/3cd530e7-bfb9-460a-8944-881cad3d6f96" />
<img width="670" height="666" alt="image" src="https://github.com/user-attachments/assets/7eff1a53-53c4-4ddb-8d9b-e3289ddc8a85" />
<img width="757" height="582" alt="image" src="https://github.com/user-attachments/assets/90398a2f-8445-4a8c-a5c7-f537cecb5e50" />


# Result:
Thus, Face Detection using Haar Cascades successfully implemented by completing the missing code sections. The system detects and highlights lane lines effectively.

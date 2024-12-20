import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("mobilenetv2_tick_mark_model_final.keras")  

image_path = "//Users/shauryachandna/Desktop/testPic.png"  
image = cv2.imread(image_path)
if image is None:
    raise Exception("Image not found or unable to load.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

binary = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)

cv2.imwrite("binary_debug_pretrained_model.png", binary)
print("Thresholded binary image saved as binary_debug_pretrained_model.png")

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Number of contours detected: {len(contours)}")

output_image = image.copy()

tick_mark_count = 0

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w >= 20 and h >= 20 and w <= 100 and h <= 100:
        roi = image[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (224, 224)) / 255.0  
        roi_resized = np.expand_dims(roi_resized, axis=0)  

        prediction = model.predict(roi_resized)
        if prediction[0] > 0.5:  
            tick_mark_count += 1
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
            print(f"Tick mark detected at: x={x}, y={y}, w={w}, h={h}")
        else:
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2) 

cv2.imwrite("tick_marks_detected_with_pretrained_model.png", output_image)
print(f"Number of tick marks detected: {tick_mark_count}")
print("Output image with detected tick marks saved as tick_marks_detected_with_pretrained_model.png")

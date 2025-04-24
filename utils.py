
import cv2
import numpy as np
import pytesseract

def reveal_text_under_blue(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    image[mask > 0] = [255, 255, 255]
    return image

def extract_text_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

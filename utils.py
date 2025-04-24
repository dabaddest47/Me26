import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance

def reveal_text_under_blue(image_rgb: np.ndarray) -> np.ndarray:
    """
    Reveal black text hidden under blue shading by:
      1) deriving a “deblue” intensity from min(R,G) on blue pixels
      2) overlaying Canny edges to sharpen text strokes
    """
    # Convert to BGR for OpenCV
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 1) Blue mask
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Split RGB channels
    r, g, b = cv2.split(image_rgb)

    # Compute “deblue” by taking min(r, g) wherever blue mask is set
    deblue = np.maximum(r, g)  # start with brighter channel for non-blue
    deblue[mask_blue > 0] = np.minimum(r[mask_blue > 0], g[mask_blue > 0])

    # Convert to PIL for contrast enhancement
    pil = Image.fromarray(deblue)
    enhancer = ImageEnhance.Contrast(pil)
    high_contrast = enhancer.enhance(3.0)  # bump contrast

    # Convert back to numpy array
    arr = np.array(high_contrast)

    # 2) Edge detection on original grayscale to capture text strokes
    gray_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_orig, 50, 150)
    # Dilate edges so they become fuller
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Overlay edges onto the deblue image
    # wherever an edge exists, set pixel to black (0)
    arr[edges > 0] = 0

    # Finally, binarize to clean up
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_from_image(gray_image: np.ndarray) -> str:
    """Perform OCR on the cleaned, binarized image."""
    return pytesseract.image_to_string(gray_image, config="--psm 6").strip()

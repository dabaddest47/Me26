import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance

def reveal_text_under_blue(image_rgb: np.ndarray) -> np.ndarray:
    """
    Suppress blue shading while preserving underlying text.
    Input: image in RGB (uint8 numpy array).
    Output: grayscale uint8 array optimized for OCR.
    """
    # Split channels
    r, g, b = cv2.split(image_rgb)

    # Compute a “de-blue” channel by taking max(r, g) at each pixel
    deblue = np.maximum(r, g)

    # Smooth out any remaining blue residue:
    # wherever blue >> deblue, pull down the blue influence
    mask = b > deblue
    deblue[mask] = ((r[mask].astype(int) + g[mask].astype(int)) // 2).astype(np.uint8)

    # Convert to PIL for contrast enhancement
    pil = Image.fromarray(deblue)
    enhancer = ImageEnhance.Contrast(pil)
    high_contrast = enhancer.enhance(2.0)

    # Back to numpy for OCR thresholding
    arr = np.array(high_contrast)
    # Binarize with Otsu to accentuate text vs. background
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

def extract_text_from_image(gray_image: np.ndarray) -> str:
    """
    Runs Tesseract OCR on a pre-thresholded grayscale image.
    """
    return pytesseract.image_to_string(gray_image, config='--psm 6').strip()

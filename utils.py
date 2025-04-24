import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance

def reveal_text_under_blue(image_rgb: np.ndarray) -> np.ndarray:
    """
    Subtracts blue channel from pixels, preserves underlying red/green text strokes,
    then enhances contrast + edges for OCR.
    Input: RGB image as uint8 numpy array.
    Output: binarized grayscale ready for Tesseract.
    """
    # Split channels
    r, g, b = cv2.split(image_rgb)

    # Subtract blue influence from red+green composite
    # We weight red/green equally
    rg = cv2.addWeighted(r, 0.5, g, 0.5, 0)
    # Remove blue: wherever blue > 0, subtract it from the composite
    composite = cv2.subtract(rg, b)

    # Clip to [0,255]
    composite = np.clip(composite, 0, 255).astype(np.uint8)

    # Convert to PIL to boost contrast
    pil = Image.fromarray(composite)
    enhancer = ImageEnhance.Contrast(pil)
    high_contrast = enhancer.enhance(2.5)  # stronger contrast

    # Convert back to numpy for edge sharpening and binarization
    arr = np.array(high_contrast)
    # Sharpen via unsharp mask
    blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(arr, 1.5, blurred, -0.5, 0)

    # Finally, Otsu binarization to isolate text
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_from_image(gray_image: np.ndarray) -> str:
    """
    Performs OCR on a clean binary image.
    """
    # Tell Tesseract to treat the image as a single uniform block of text
    return pytesseract.image_to_string(gray_image, config="--psm 6").strip()

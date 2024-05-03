import cv2
import pytesseract
import numpy as np

# Path to Tesseract executable
#pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


# Load pre-trained cascade classifier for license plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Function to detect license plates
def detect_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80))
    for (x, y, w, h) in plates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi = img[y:y + h, x:x + w]
        text = recognize_characters(roi)
        print("Recognized text:", text)
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Function to recognize characters using Tesseract OCR
def recognize_characters(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply image processing techniques here if needed
    text = pytesseract.image_to_string(gray, config='--psm 8 --oem 3')
    return text

# Capture video stream from camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    detect_plate(frame)
    
    cv2.imshow('License Plate Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

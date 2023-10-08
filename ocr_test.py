# import easyocr
# reader = easyocr.Reader(['en'])

# result = reader.readtext('plates\Car_plate_detectorframe1.png')
# print(result)


# import pytesseract
from PIL import Image
# pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'
# print(pytesseract.image_to_string('plates\Car_plate_detectorframe0.png', timeout=2))

from paddleocr import PaddleOCR
import numpy as np

ocr = PaddleOCR(use_angle_cls=False, lang="en")
i = Image.open("images\Car_plate_detectorframe0.png")
result = ocr.ocr(np.asarray(i) , cls=True)
print(result)
print(result[0][0][1][0])


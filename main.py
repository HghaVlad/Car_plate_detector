import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
from torch import Tensor


YOLO_MODEL_PT = "yolosmall7.pt"

ORGINAL_IMG_PATH = "mile.png"
RESULT_PATH = r"E:\\Code works\Data Science\\MIEM_carplate_detector\\Car_plate_detector\\images"


def init_yolo() -> YOLO:
    model = YOLO(YOLO_MODEL_PT)
    print("Model is ready")
    return model


def get_yolo_boxes(image: Image, model: YOLO) -> Tensor:
    result = model.predict(image, verbose=False)
    boxes = result[0].boxes.data
    print(f"Found {len(boxes)} car plates")
    return boxes


def crop_image(image: Image, boxes: Tensor):
    car_plates = []
    for box in boxes:
        x1, y1, x2, y2 = map(lambda x: round(x.item()), box[:4])
        coef = round(box[4].item(), 3)
        cropped_image = image.crop((x1, y1, x2, y2))
        car_plates.append([cropped_image, coef])
    
    return car_plates


def draw_box(boxes: list, lables: list, image: Image) -> Image:
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, lables):
        x1, y1, x2, y2 = map(lambda x: round(x.item()), box[:4])
        draw.rectangle((x1, y1, x2, y2))
        draw.text((x1, y1-20), str(label), fill="red")
    return image


def init_paddleocr():
    ocr = PaddleOCR(use_angle_cls=False, lang="en")
    return ocr

def recognise_text(image, model):
    result = model.ocr(np.asarray(image) , cls=True)
    print(result)
    if result[0] is None:
        return "NOT DETECTED"
    
    return result[0][0][1][0]




if __name__ == "__main__":
    detect_model = init_yolo()
    ocr_model = init_paddleocr()
    current_image = Image.open(ORGINAL_IMG_PATH)
    boxes = get_yolo_boxes(current_image, detect_model)
    
    plates = crop_image(current_image, boxes)
    labels = []
    for i, plate in enumerate(plates):
        plate[0].save(f"{RESULT_PATH}\\frame{i}.png")
        label = recognise_text(plate[0], ocr_model)
        labels.append(label)
    
    output_image = draw_box(boxes, labels, current_image)
    output_image.save(RESULT_PATH + "\\frame_end.png")
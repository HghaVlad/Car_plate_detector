import cv2
from main import init_yolo, init_paddleocr, get_yolo_boxes, crop_image, recognise_text, draw_box
import numpy as np
from PIL import Image

CAMERA_NUMBER = 0


def get_camera_frames():
    camera = cv2.VideoCapture(CAMERA_NUMBER) 
    print("Camera was opened")
    while True:
        ret, frame = camera.read()       

        yield frame
    camera.release()


def main():
    detect_model = init_yolo()
    ocr_model = init_paddleocr()

    for frame in get_camera_frames():
        
        current_image = Image.fromarray(frame)
        
        boxes = get_yolo_boxes(current_image, detect_model)
        if len(boxes) > 0:
            labels = []
            plates = crop_image(current_image, boxes)
            for plate in plates:
                label = recognise_text(plate[0], ocr_model)
                labels.append(label)

            output_frame = np.array(draw_box(boxes, labels, current_image))
        else:
            output_frame = frame

        cv2.imshow("Press ESC or Enter to close", output_frame)

        k = cv2.waitKey(1)
        if k%256 in [27, 13]:
            # ESC pressed
            print("Escape hit, closing...")
            break

main()
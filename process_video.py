import cv2
from main import init_yolo, init_paddleocr, get_yolo_boxes, crop_image, recognise_text, draw_box
import numpy as np
from PIL import Image

VIDEO_PATH = "E:\Code works\Data Science\MIEM_carplate_detector\\test2.mp4"
OUTPUT_PATH = "E:\Code works\Data Science\MIEM_carplate_detector\\new.mkv"
OUTPUT_VIDEO_SIZES = (720, 480) # (Width, Height) e. (720, 480)


def get_video_frames():
    camera = cv2.VideoCapture(VIDEO_PATH) 

    while True:
        ret, frame = camera.read()    
        if ret:   
            yield frame


def main():
    detect_model = init_yolo()
    ocr_model = init_paddleocr()

    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, OUTPUT_VIDEO_SIZES)

    for frame in get_video_frames():
        
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
        out.write(output_frame)

        k = cv2.waitKey(1)
        if k%256 in [27, 13]:
            # ESC pressed
            print("Escape hit, closing...")
            print("Video hasn't been saved")
            break
    
    out.release()
    print(f"Video was saved to {OUTPUT_PATH}")
            
main()
from ultralytics import YOLO
import cv2
import math

objects = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "bag", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("yolo-Weights/yolov8n.pt")

while True:
    ret, img= cap.read()

    results = model(img, stream=True)
    for result in results:
        boxes = result.boxes

        for box in boxes:
            # Calculate confidence level
            confidence = math.ceil((box.conf[0] * 100)) / 100

            if confidence > 0.6:
                # Set up bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Overlay bounding box on webcam frame
                cv2.rectangle(img, (x1, y1), (x2, y2), (31, 43, 234), 3)

                # Object name
                object = int(box.cls[0])

                # Object details
                text_origin = [x1, y1]
                text_font = cv2.FONT_HERSHEY_SIMPLEX
                text_font_scale = 1
                text_color = (83, 70, 38)
                text_thickness = 2

                cv2.putText(img, objects[object], text_origin, text_font,
                            text_font_scale, text_color, text_thickness)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

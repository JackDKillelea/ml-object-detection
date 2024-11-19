from ultralytics import YOLO
import cv2
import math

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "bag", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    ret, img= cap.read()

    results = model(img, stream=True)
    for result in results:
        boxes = result.boxes

        for box in boxes:
            # Calculate confidence level
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            if confidence > 1:
                # Set up bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Overlay bounding box on webcam frame
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





# # Load a pre-trained object detection model (e.g., YOLOv8)
# model = YOLO("my_yolo11n.pt")
#
# # Train the model
# train_results = model.train(
#     data="coco8.yaml",  # path to dataset YAML
#     epochs=100,  # number of training epochs
#     imgsz=640,  # training image size
#     device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
# )
# # Perform object detection on an image
# results = model("me.jpg")
# results[0].show()
#
# # Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model












# model = tf.saved_model.load('my_yolo11n.pt')
# # model = onnxruntime.InferenceSession("best.onnx")
#
# input_name = model.get_inputs()[0].name
# output_names = [output.name for output in model.get_outputs()]
#
# # Initialize the webcam
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#
#     # Preprocess the frame (resize, normalize, etc.)
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     image_np = np.array(image)
#     input_tensor = tf.convert_to_tensor(image_np)
#     input_tensor = input_tensor[tf.newaxis]  # Add batch dimension
#
#     # Run inference
#     detections = model(input_tensor)
#
#     # Postprocess the output (extract bounding boxes, class labels, etc.)
#     # ... (Implement postprocessing based on your model's output format)
#
#     # Draw bounding boxes and labels on the frame
#     # ... (Use OpenCV functions to draw)
#
#     cv2.imshow('Object Detection', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
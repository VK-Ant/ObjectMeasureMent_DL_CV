import cv2
import numpy as np

# Load COCO classes
classes = []
with open(r"coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load YOLOv3 model
# Replace with actual paths
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Get output layer names
output_layer_names = net.getUnconnectedOutLayersNames()

# Known reference object width in pixels and centimeters
# Replace with actual width of reference object in pixels
reference_object_width_px = 100
# Replace with actual width of reference object in cm
reference_object_width_cm = 10

target_class_name = "cup"
target_class_id = classes.index(target_class_name)

webcam = True
path = r'/home/vk/Desktop/ObjectMeasureMent_DL_CV-main/ObjectMeasureMent_DL_CV/1stmethod_OpenCV/images (1).jpeg'

# Start webcam capture
cap = cv2.VideoCapture(2)  # 0 represents the default camera

while cap.isOpened():


    if webcam:
        ret, frame = cap.read()
    else:
        frame = cv2.imread(path)



    height, width, _ = frame.shape

    # Process frame for object detection
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Perform object detection
    outs = net.forward(output_layer_names)

    target_detected = False

    # Loop through detected objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == target_class_id:  # Adjust threshold and target class as needed
                target_detected = True
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate object width and height in cm
                object_width_cm = (
                    w / reference_object_width_px) * reference_object_width_cm
                object_height_cm = (
                    h / reference_object_width_px) * reference_object_width_cm

                # Draw bounding box and label on frame
                color = (0, 255, 0)  # Green color for bounding box
                cv2.rectangle(frame, (center_x - w // 2, center_y - h // 2),
                              (center_x + w // 2, center_y + h // 2), color, 2)
                label = f"{classes[class_id]} - Width: {object_width_cm:.2f} cm, Height: {object_height_cm:.2f} cm"
                cv2.putText(frame, label, (center_x - w // 2, center_y -
                            h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display message if target not detected
    if not target_detected:
        cv2.putText(frame, "Target not detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

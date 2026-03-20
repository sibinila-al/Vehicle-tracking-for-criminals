# Import All the Required Libraries
import json
import cv2
from ultralytics import YOLO
import numpy as np
import math
import re
import os
import mysql.connector
from datetime import datetime
from paddleocr import PaddleOCR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Create a Video Capture Object for the webcam
cap = cv2.VideoCapture(0)

# Initialize the YOLO Model
model = YOLO("weights/best.pt")

# Print model classes (for debugging)
print("Model Classes:", model.names)

# Initialize the frame count
count = 0

# Class Names
className = ["License"]

# Initialize the Paddle OCR
ocr = PaddleOCR(use_textline_orientation=True)


def paddle_ocr(frame, x1, y1, x2, y2):

    if y2 <= y1 or x2 <= x1:
        return ""

    frame = frame[y1:y2, x1:x2]

    result = ocr.ocr(frame)
    print(result)
    if not result or result[0] is None:
        return ""

    text = ""

    for line in result:
          if line is None:
              continue
          for box, (t, s) in line:
              if s > 0.6:
                  text = t
    pattern = re.compile(r'[\W]')
    text = pattern.sub('', text)
    text = text.replace("???", "")
    text = text.replace("O", "0")
    text = text.replace("粤", "")

    return str(text)


def save_json(license_plates, startTime, endTime):

    interval_data = {
        "Start Time": startTime.strftime("%d-%m-%Y %I:%M:%S %p"),
        "End Time": endTime.strftime("%d-%m-%Y %I:%M:%S %p"),
        "License Plate": list(license_plates)
         
    }

    interval_file_path = "json/output_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json"

    with open(interval_file_path, 'w') as f:
        json.dump(interval_data, f, indent=2)

    cummulative_file_path = "json/LicensePlateData.json"

    if os.path.exists(cummulative_file_path):
        with open(cummulative_file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(interval_data)

    with open(cummulative_file_path, 'w') as f:
        json.dump(existing_data, f, indent=2)

    save_to_database(license_plates, startTime, endTime)


def save_to_database(license_plates, start_time, end_time):

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="projectx"
    )

    cursor = conn.cursor()

    for plate in license_plates:

        ret2, frame2 = cap.read()

        if ret2:
            _, buffer = cv2.imencode('.jpg', frame2)
            image = buffer.tobytes()

            cursor.execute('''
                INSERT INTO licenseplates(start_time, end_time, license_plate, image)
                VALUES (%s, %s, %s, %s)
            ''', (startTime.strftime("%d-%m-%Y %I:%M:%S %p"), end_time.strftime("%d-%m-%Y %I:%M:%S %p"), plate, image))
    print("============================================")
    print(plate)
    print("============================================")
    conn.commit()
    conn.close()


startTime = datetime.now()
license_plates = set()

while True:

    ret, frame = cap.read()

    if ret:

        currentTime = datetime.now()
        count += 1
        print(f"Frame Number: {count}")

        results = model.predict(frame, conf=0.15)

        print("Detections:", len(results[0].boxes))

        for result in results:

            boxes = result.boxes

            for box in boxes:

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                classNameInt = int(box.cls[0])

                if classNameInt < len(className):
                    clsName = className[classNameInt]
                else:
                    clsName = "License"

                conf = math.ceil(box.conf[0] * 100) / 100

                label = paddle_ocr(frame, x1, y1, x2, y2)

                if label:
                    license_plates.add(label)

                textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + textSize[0], y1 - textSize[1] - 3

                cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)

                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5,
                            [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        if (currentTime - startTime).seconds >= 20:

            endTime = currentTime
            save_json(license_plates, startTime, endTime)

            startTime = currentTime
            license_plates.clear()

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('1'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
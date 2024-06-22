#!/usr/bin/env python

from ctypes.wintypes import WORD
import rospy
import pandas as pd
import cv2
from ultralytics import YOLO
from PIL import Image
from classifier import predict
from std_msgs.msg import String
import json
from dotenv import load_dotenv

load_dotenv()

WORD_DIR = "/home/mustar/car_ws/src/car_assistant/data"
detector = YOLO('yolov8n.pt')

class CarRecognition:
    def __init__(self, pubsliherNodeName, topicName) -> None:
        self.publisherNodeName = pubsliherNodeName
        rospy.init_node(self.publisherNodeName)
        self.car_data = pd.read_csv(f'{WORD_DIR}/car_data.csv')
        self.model = YOLO(f'{WORD_DIR}/model/best.pt')
        self.pub = rospy.Publisher(topicName, String, queue_size=60)

    def recognition(self, image):
        detection_results = detector(image, classes=[2, 3, 5, 7])
        cars_info = []

        for res in detection_results:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.numpy()[0])
                roi = image[y1:y2, x1:x2]
                pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

                result = predict(pil_roi)
                car_info = self.car_data.loc[self.car_data['model_name'] == result].values
                cars_info.append((x1, y1, x2, y2, result, car_info))

        return cars_info

    def run(self):
        cars_object = []
        cap = cv2.VideoCapture(0)
        

        if not cap.isOpened():
            rospy.logwarn("Warn: Could not open webcam.")

        while True:
            _, frame = cap.read()
            cars_info = self.recognition(frame)
            cars_object = []

            for (x1, y1, x2, y2, result, car_info) in cars_info:
                cars_object.append(str(car_info))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, result, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            delimiter = ' || '
            concanated_string = delimiter.join(cars_object)
            self.pub.publish(String(concanated_string))

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    car_detector = CarRecognition("car_recognition", "car_recognition_output")
    car_detector.run()

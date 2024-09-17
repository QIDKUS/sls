from flask import Flask, Response, request
from ultralytics import YOLO
import numpy as np
import imutils
import cv2

# Import parameters
from config.parameters import WIDTH, start_status, shoplifting_status, not_shoplifting_status
from config.parameters import cls0_rect_color, cls1_rect_color, conf_color, status_color
from config.parameters import quit_key, frame_name

app = Flask(__name__)

# Load model
mymodel = YOLO("models/live_weights.pt")

def process_frame(frame):
    frame = imutils.resize(frame, width=WIDTH)
    result = mymodel.predict(frame)
    cc_data = np.array(result[0].boxes.data)

    status = start_status

    if len(cc_data) != 0:
        xywh = np.array(result[0].boxes.xywh).astype("int32")
        xyxy = np.array(result[0].boxes.xyxy).astype("int32")
        
        for (x1, y1, _, _), (_, _, w, h), (_, _, _, _, conf, clas) in zip(xyxy, xywh, cc_data):
            if clas == 1:
                cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), cls1_rect_color, 2)
                half_w = w / 2
                x = int(half_w + x1)
                cv2.circle(frame, (x, y1), 6, (0, 0, 255), 8)

                text = "{}%".format(np.round(conf * 100, 2))
                cv2.putText(frame, text, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 2)
                status = shoplifting_status
        
        if status == shoplifting_status:
            cv2.putText(frame, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    return frame
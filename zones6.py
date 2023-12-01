#Código fodido detecta pior, mais lento (em média 6 seg em relação ao vídeo original)
import numpy as np
import cv2
import json
import time
import paho.mqtt.client as mqtt
from datetime import datetime
from models import TRTModule
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox
import torch


class CountObject:
    def __init__(self, json_path):
        # Carrega o modelo TensorRT
        self.engine_path = 'modelos/yolov8s.engine'
        self.device_name = 'cuda:0'
        self.device = torch.device(self.device_name)
        self.Engine = TRTModule(self.engine_path, self.device)
        self.W, self.H = 640, 640
        self.Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.display_width = 1100
        self.display_height = None
        self.last_saved_time = time.time()

        self.client = mqtt.Client("Your_Client_Name")
        self.client.connect("localhost", 1883, 60)

        self.zones = self.load_zones(json_path)

    def load_zones(self, zones_path):
        with open(zones_path, 'r') as file:
            zones_data = json.load(file)
        return [np.array(points, dtype=np.int32) for points in zones_data.values() if len(points) >= 3]

    def process_frame(self, frame: np.ndarray, i) -> np.ndarray:
        frame_resized, ratio, dwdh = letterbox(frame, (self.W, self.H))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        tensor = blob(frame_rgb, return_seg=False)
        dwdh = torch.tensor(dwdh * 2, dtype=torch.float32, device=self.device)
        tensor = torch.tensor(tensor, device=self.device)

        # Inferência com TensorRT
        data = self.Engine(tensor)
        bboxes, scores, labels = det_postprocess(data)
        bboxes -= dwdh
        bboxes /= ratio

        # Filtra e processa detecções na GPU
        detections_filtered = [(bbox, score, label) for bbox, score, label in zip(bboxes, scores, labels) if label == 0 and score > 0.5]

        count_people = 0
        for polygon in self.zones:
            cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)

            # Calcula o retângulo delimitador para o polígono
            area_xmin, area_ymin = np.min(polygon, axis=0)
            area_xmax, area_ymax = np.max(polygon, axis=0)
            area = (area_xmin, area_ymin, area_xmax, area_ymax)

            for bbox, score, label in detections_filtered:
                if self.bbox_intersects_area(bbox, area):
                    count_people += 1
                    bbox = bbox.cpu().round().int().tolist()
                    cv2.rectangle(frame, bbox[:2], bbox[2:], self.colors[int(label)], 2)

        cv2.putText(frame, f'Pessoas: {count_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        current_time = datetime.now().strftime("%H:%M:%S")
        if time.time() - self.last_saved_time >= 3:
            payload = json.dumps({'Time': current_time, 'Count': count_people})
            self.client.publish("sensor/data", payload)
            self.last_saved_time = time.time()

        if self.display_height is None:
            self.display_height = self.display_width * frame.shape[0] // frame.shape[1]

        resized_frame = cv2.resize(frame, (self.display_width, self.display_height), interpolation=cv2.INTER_AREA)
        cv2.imshow('Processed Video', resized_frame)

        return frame

    def bbox_intersects_area(self, bbox, area):
        """ Verifica se a caixa delimitadora intersecta com a área de monitoramento """
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox
        area_xmin, area_ymin, area_xmax, area_ymax = area

        return not (bbox_xmax < area_xmin or bbox_xmin > area_xmax or 
                    bbox_ymax < area_ymin or bbox_ymin > area_ymax)

    def run(self, input_video_path, output_video_path):
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print("Não foi possível abrir o stream de vídeo.")
            return

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Não foi possível ler o frame do vídeo.")
                break

            processed_frame = self.process_frame(frame, 0)
            out.write(processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.client.disconnect()

if __name__ == '__main__':
    json_path = 'modelos/zones.json'
    count_object = CountObject(json_path)
    count_object.run('data/pessoas1.mp4', 'teste.mp4')

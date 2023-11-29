from models import TRTModule  # isort:skip
import time
import cv2
import torch

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox

def main() -> None:
    # Definição estática das variáveis anteriormente obtidas por argparse
    engine_path = 'modelos/yolov8s.engine'  # Substitua com o caminho correto
    video_path = 'data/pessoas1.mp4'  # Substitua com o caminho correto
    show_results = True  # Exibir resultados da detecção
    #output_directory = './output'  # Diretório de saída
    device_name = 'cuda:0'  # Dispositivo para inferência do TensorRT

    device = torch.device(device_name)
    Engine = TRTModule(engine_path, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        draw = frame.copy()
        frame, ratio, dwdh = letterbox(frame, (W, H))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = blob(frame_rgb, return_seg=False)
        dwdh = torch.tensor(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.tensor(tensor, device=device)

        # inference
        data = Engine(tensor)

        bboxes, scores, labels = det_postprocess(data)
        if bboxes.numel() == 0:
            print('Nenhum objeto detectado no frame atual!')
            continue
        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw, f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)

        if show_results:
            cv2.imshow('result', draw)
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

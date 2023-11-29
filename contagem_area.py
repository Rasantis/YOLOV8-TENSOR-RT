from models import TRTModule
import cv2
import torch
from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox

def main() -> None:
    # Configurações do modelo e do vídeo
    engine_path = 'modelos/yolov8s.engine'  # Caminho para o arquivo de engine
    video_path = 'data/pessoas1.mp4'  # Caminho para o vídeo
    show_results = True  # Defina como True para mostrar o resultado em uma janela
    device_name = 'cuda:0'  # Dispositivo para inferência do TensorRT

    # Configuração do dispositivo e do modelo
    device = torch.device(device_name)
    Engine = TRTModule(engine_path, device)
    H, W = Engine.inp_info[0].shape[-2:]
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    # Abre o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo {video_path}")
        return

    # Aumentar a área de monitoramento (xmin, ymin, xmax, ymax)
    monitor_area = (100, 100, 600, 600)  # Ajuste conforme necessário
    monitor_area_color = (0, 255, 0)  # Cor verde para a área de monitoramento

    # Definir o tamanho da janela exibida
    display_size = (640, 480)  # Largura x Altura

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

        # Inferência
        data = Engine(tensor)
        bboxes, scores, labels = det_postprocess(data)
        if bboxes.numel() == 0:
            print('Nenhum objeto detectado no frame atual!')
            continue
        bboxes -= dwdh
        bboxes /= ratio

        # Desenha a área de monitoramento
        cv2.rectangle(draw, (monitor_area[0], monitor_area[1]), (monitor_area[2], monitor_area[3]), monitor_area_color, 2)

        # Contagem de pessoas na área de monitoramento
        count_people = 0
        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw, f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)

            # Verificar se a detecção está dentro da área de monitoramento
            if (label == 0 or CLASSES[label] == 'person') and bbox_intersects_area(bbox, monitor_area):
                count_people += 1

        # Exibir contagem na área de monitoramento
        text = f'Pessoas: {count_people}'
        text_location = (10, 30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        font_thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_width, text_height = text_size

        # Desenha o retângulo cinza como fundo para o texto
        cv2.rectangle(draw, (text_location[0], text_location[1] - text_height - 10),
                    (text_location[0] + text_width + 10, text_location[1]),
                    (192, 192, 192), -1)

        # Exibe o texto sobre o retângulo
        cv2.putText(draw, text, text_location, font, font_scale, font_color, font_thickness)

        # Redimensionar o frame para a janela exibida
        resized_frame = cv2.resize(draw, display_size)

        # Mostra o resultado
        if show_results:
            cv2.imshow('result', resized_frame)
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def bbox_intersects_area(bbox, area):
    """ Verifica se a caixa delimitadora intersecta com a área de monitoramento """
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox
    area_xmin, area_ymin, area_xmax, area_ymax = area

    return not (bbox_xmax < area_xmin or bbox_xmin > area_xmax or 
                bbox_ymax < area_ymin or bbox_ymin > area_ymax)

if __name__ == '__main__':
    main()

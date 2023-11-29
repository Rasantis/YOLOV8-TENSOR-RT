Comando para exportar o modelo de .pt para onnx:

```
python export-det.py --weights yolov8s.pt --iou-thres 0.65 --conf-thres 0.25 --topk 100 --opset 11 --sim --input-shape 1 3 640 640 --device cuda:0
```

Argumentos usados e explicação:

1. `--weights yolov8s.pt`: Especifica o caminho para o arquivo de pesos do modelo YOLOv8 que será usado para exportar o modelo ONNX.
2. `--iou-thres 0.65`: Define o limiar de Intersecção sobre União (IoU) para a Non-Maximum Suppression (NMS) em 0.65. Isso é usado para filtrar caixas delimitadoras com base na sobreposição.
3. `--conf-thres 0.25`: Define o limiar de confiança para a detecção em 0.25. Detecções com confiança abaixo deste limiar serão descartadas.
4. `--topk 100`: Especifica o número máximo de detecções a serem consideradas após a aplicação do limiar de confiança e IoU (100, neste caso).
5. `--opset 11`: Define a versão do conjunto de operações ONNX a ser usada para exportação do modelo.
6. `--sim`: Indica que o modelo deve ser simplificado. Isso geralmente envolve a remoção de operações redundantes do modelo ONNX para otimização.
7. `--input-shape 1 3 640 640`: Define a forma de entrada para o modelo. Neste caso, representa um lote de tamanho 1 (batch size), com 3 canais de cor, e dimensão espacial de 640x640.
8. `--device cuda:0`: Especifica que o dispositivo CUDA (normalmente uma GPU) com ID 0 deve ser usado para a operação de exportação.

Comando para exportar de .pt para .engine (formato do TENSOR RT):

```
python build.py --weights yolov8s.onnx --iou-thres 0.65 --conf-thres 0.25 --topk 100 --fp16 --device cuda:0
```

1. `--weights yolov8s.onnx`: Caminho para o arquivo ONNX do modelo YOLOv8. Este é o modelo que será convertido para o formato TensorRT.
2. `--iou-thres 0.65`: Define o limiar de Intersecção sobre União (IoU) para o plugin de Supressão Máxima Não Máxima (NMS) em 0.65.
3. `--conf-thres 0.25`: Define o limiar de confiança para o plugin de NMS em 0.25.
4. `--topk 100`: Especifica o número máximo de caixas delimitadoras (bounding boxes) a serem consideradas (100 neste caso).
5. `--fp16`: Indica que o motor deve ser construído no modo de precisão mista FP16, o que pode melhorar o desempenho em GPUs compatíveis.
6. `--device cuda:0`: Especifica que o dispositivo CUDA (normalmente uma GPU) com ID 0 deve ser usado para a construção do motor TensorRT.

Comando para rodar a inferencia usando o Tensor RT:

```
python inferencia_rt.py --engine modelos/yolov8s.engine --imgs data --show --out-dir outputs --device cuda:0
```

1. `--engine yolov8s.engine`: Especifica o caminho para o arquivo do motor (engine) TensorRT. Este arquivo contém o modelo YOLOv8 otimizado para inferência rápida.
2. `--imgs data`: Indica o caminho para o vídeo no qual a inferência será realizada. Neste caso, espera-se que o vídeo esteja localizado no diretório `data`.
3. `--show`: Se presente, o script exibirá os resultados da detecção em uma janela de visualização em tempo real.
4. `--out-dir outputs`: Define o diretório onde os resultados da inferência (se houver alguma saída específica do script, como imagens processadas) serão salvos. Por padrão, aponta para o diretório `./output`.
5. `--device cuda:0`: Especifica que a inferência deve ser realizada no dispositivo CUDA (uma GPU, neste caso) com ID 0.


CONSUMO DE MEMÓRIA DE GPU USANDO YOLOV8S É DE 225 MB

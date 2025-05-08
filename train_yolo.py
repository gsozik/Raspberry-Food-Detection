from ultralytics import YOLO

if __name__ == "__main__":
    # 1) Загружаем лёгкую предобученную модель (можно поменять на yolov8s.pt или вашу кастомную)
    model = YOLO('yolov8n.pt')
    # 2) Запускаем обучение
    model.train(
        data='data.yaml',      # путь к data.yaml, который вы сгенерировали
        epochs=50,             # количество эпох (можете увеличить)
        imgsz=640,             # размер входных картинок
        batch=16,              # размер батча (если мало RAM, уменьшите)
        name='uecfoodpix-exp'  # имя эксперимента (создаст runs/detect/uecfoodpix-exp)
    )
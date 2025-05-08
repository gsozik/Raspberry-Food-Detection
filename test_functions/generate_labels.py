import os
import cv2
import numpy as np

# -------------- Настройка путей --------------
# Папка, где лежит этот скрипт:
SCRIPT_DIR = os.path.dirname(__file__)
# Корень ML-данных:
DATA_DIR   = os.path.join(SCRIPT_DIR, 'testML', 'data')
# Внутри неё — папка с классами:
BASE       = os.path.join(DATA_DIR, 'UECFoodPIXCOMPLETE')
# Файл с названиями классов находится параллельно:
CAT_FILE   = os.path.join(DATA_DIR, 'category.txt')

# Проверяем, что всё на месте
if not os.path.isdir(BASE):
    raise FileNotFoundError(f"Не найдена папка с картинками и масками: {BASE}")
if not os.path.isfile(CAT_FILE):
    raise FileNotFoundError(f"Не найден файл category.txt: {CAT_FILE}")

# -------------- Читаем классы --------------
classes = [l.strip() for l in open(CAT_FILE, encoding='utf-8')]

# -------------- Описываем разбиение --------------
SPLITS = {
    'train': {'img':'train/img', 'mask':'train/mask', 'labels':'train/labels'},
    'val':   {'img':'test/img',  'mask':'test/mask',  'labels':'test/labels'}
}

# -------------- Генерация аннотаций --------------
for split, paths in SPLITS.items():
    img_dir  = os.path.join(BASE, paths['img'])
    mask_dir = os.path.join(BASE, paths['mask'])
    lbl_dir  = os.path.join(BASE, paths['labels'])
    os.makedirs(lbl_dir, exist_ok=True)

    for fname in os.listdir(img_dir):
        if not fname.lower().endswith('.jpg'):
            continue

        img_path  = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname.replace('.jpg', '.png'))
        img  = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  Пропускаем без маски: {mask_path}")
            continue

        h, w = img.shape[:2]
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            print(f"  Нет ненулевых пикселей в маске: {mask_path}")
            continue

        # Границы объекта
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        # Нормализованные координаты
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        # Класс — значение пикселя минус 1 (в маске все пиксели одинаковы)
        class_id = int(mask[ys[0], xs[0]]) - 1

        # Записываем в YOLO .txt
        out_txt = os.path.join(lbl_dir, fname.replace('.jpg', '.txt'))
        with open(out_txt, 'w') as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"Аннотации для '{split}' созданы в {lbl_dir}")

print("Генерация YOLO-меток завершена.")
import cv2
import numpy as np
import os
import cv2
import numpy as np

def detect_blobs(mask: np.ndarray,
                 frame: np.ndarray,
                 min_area: int = 500,
                 min_width: int = 20,
                 min_height: int = 20) -> list[dict]:
    """
    Находит блобы на бинарной маске и возвращает их ROI из исходного кадра.
    Отфильтровывает шум по минимальной площади и минимальным размерам рамки.

    Args:
        mask: бинарная маска (0—фон, 255—объекты).
        frame: оригинальный цветной кадр (BGR).
        min_area: минимальная площадь контура для отбора.
        min_width: минимальная ширина bounding box.
        min_height: минимальная высота bounding box.

    Returns:
        List[dict], где каждый dict содержит:
            'bbox': (x, y, w, h)
            'roi' : np.ndarray — вырезанный фрагмент из оригинального кадра.
    """
    # 1) Поиск контуров на маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # отсекаем слишком узкие или низкие рамки
        if w < min_width or h < min_height:
            continue
        # вырезаем ROI из оригинального кадра
        roi = frame[y:y+h, x:x+w].copy()
        blobs.append({'bbox': (x, y, w, h), 'roi': roi, 'area': area})

    # сортируем по площади (крупные в начале)
    blobs.sort(key=lambda b: b['area'], reverse=True)
    return blobs

def merge_overlapping_boxes(bboxes):
    """
    bboxes: list of (x, y, w, h)
    возвращает список объединённых (x, y, w, h)
    """
    # конвертируем в [x1,y1,x2,y2]
    rects = [[x, y, x + w, y + h] for x, y, w, h in bboxes]
    merged = []
    while rects:
        x1, y1, x2, y2 = rects.pop(0)
        changed = True
        # пробегаем по остальным, сливаем, если пересекаются
        while changed:
            changed = False
            new_rects = []
            for xx1, yy1, xx2, yy2 in rects:
                # Проверка на пересечение
                if not (xx2 < x1 or xx1 > x2 or yy2 < y1 or yy1 > y2):
                    # есть перекрытие → объединяем
                    x1 = min(x1, xx1)
                    y1 = min(y1, yy1)
                    x2 = max(x2, xx2)
                    y2 = max(y2, yy2)
                    changed = True
                else:
                    new_rects.append([xx1, yy1, xx2, yy2])
            rects = new_rects
        merged.append((x1, y1, x2 - x1, y2 - y1))
    return merged



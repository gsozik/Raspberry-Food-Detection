import cv2
import numpy as np
import json

import cv2
import numpy as np

# ----------------------------------------
# Функции сегментации с учётом статичного фона
# ----------------------------------------

import cv2
import numpy as np

def segment_by_background(frame: np.ndarray,
                          background: np.ndarray,
                          thresh: int = 30,
                          open_k: int = 5,
                          close_k: int = 5,
                          merge_contours: bool = True) -> (np.ndarray, np.ndarray):
    """
    Сегментация объектов на статичном фоне с заполнением контуров.

    Если merge_contours=True, объединяет все найденные контуры в один
    выпуклый оболочкой (convex hull), чтобы получить единый объект.

    Возвращает:
        filled_mask: бинарная маска (0 или 255) с одним сплошным объектом.
        segmented: цветной кадр, в котором оставлены только объекты.
    """
    # 1) Вычитание фона + grayscale
    diff = cv2.absdiff(frame, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # 2) Пороговая бинаризация
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    # 3) Морфологическая обработка
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # 4) Поиск контуров
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, frame.copy()

    # 5) Объединяем контуры, если нужно
    if merge_contours:
        # собираем все точки из контуров и строим convex hull
        all_pts = np.vstack(contours)
        hull = cv2.convexHull(all_pts)
        final_contours = [hull]
    else:
        final_contours = contours

    # 6) Заполняем полученные контуры
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, final_contours, -1, 255, thickness=-1)

    # 7) Маскируем исходный кадр
    segmented = cv2.bitwise_and(frame, frame, mask=filled)

    return filled, segmented
def segment_by_subtraction(frame, background, diff_thresh, open_k, close_k):
    # 1) Абс. разница
    diff = cv2.absdiff(frame, background)
    # 2) В серый
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # 3) Порог
    _, mask = cv2.threshold(diff_gray, diff_thresh, 255, cv2.THRESH_BINARY)
    # 4) Открытие (удаление мелких шумов)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    # 5) Закрытие (заполнение дыр)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, k_close)
    # 6) Наложение маски
    segmented = cv2.bitwise_and(frame, frame, mask=mask_clean)
    return mask_clean, segmented





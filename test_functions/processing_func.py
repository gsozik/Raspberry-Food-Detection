import cv2
import numpy as np


def segment_image(frame, background_path='background.png', thresh=30, kernel_size=(5, 5)):
    # 1) Загрузка фона
    background = cv2.imread(background_path)
    if background is None:
        raise FileNotFoundError(f"Не найден файл фона: {background_path}")

    # 2) Подгонка размера фона под кадр
    if background.shape[:2] != frame.shape[:2]:
        background = cv2.resize(background, (frame.shape[1], frame.shape[0]))

    # 3) Вычисление среднего цвета фона (B, G, R)
    mean_color = cv2.mean(background)[:3]

    # 4) Расстояние каждого пикселя кадра до цвета фона
    diff = np.linalg.norm(frame.astype('float32') - np.array(mean_color)[None, None, :], axis=2)

    # 5) Нормализация и бинаризация по порогу
    diff_uint8 = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    _, mask = cv2.threshold(diff_uint8, thresh, 255, cv2.THRESH_BINARY)

    # 6) Морфологическая очистка (открытие)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 7) Применение маски к кадру
    segmented = cv2.bitwise_and(frame, frame, mask=mask_clean)

    return mask_clean, segmented


if __name__ == "__main__":
    print('True')

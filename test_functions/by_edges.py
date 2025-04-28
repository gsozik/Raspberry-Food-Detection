import cv2
import numpy as np

def segment_by_subtraction(frame, background_path='background.png',
                           diff_thresh=30,
                           open_kernel_size=(3,3),
                           close_kernel_size=(7,7)):

    # 1) Загрузка фона
    bg = cv2.imread(background_path)
    if bg is None:
        raise FileNotFoundError(f"Не найден фон: {background_path}")
    # 2) Приведение размеров
    if bg.shape[:2] != frame.shape[:2]:
        bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
    # 3) Абсолютная разница
    diff = cv2.absdiff(frame, bg)
    # 4) В серый
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # 5) Бинаризация
    _, mask = cv2.threshold(diff_gray, diff_thresh, 255, cv2.THRESH_BINARY)
    # 6) Открытие (эрозия→дилатация) для удаления шума
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel_size)
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    # 7) Закрытие (дилатация→эрозия) для заполнения дыр
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel_size)
    mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, close_k)
    # 8) Вырезание объекта из исходного кадра
    segmented = cv2.bitwise_and(frame, frame, mask=mask_clean)
    return mask_clean, segmented

if __name__ == "__main__":
    # загрузите ваш кадр
    frame = cv2.imread("../frames/snapshot.png")
    if frame is None:
        print("Не найден файл snapshot.png")
        exit(1)

    mask, seg = segment_by_subtraction(frame,
                                       background_path="../frames/background.png",
                                       diff_thresh=30,
                                       open_kernel_size=(3,3),
                                       close_kernel_size=(7,7))

    # сохранить результаты
    cv2.imwrite("../mask_diff.png", mask)
    cv2.imwrite("../seg_diff.png", seg)
    print("Готово: mask_diff.png, seg_diff.png")

    # показать
    cv2.imshow("Mask (diff)", mask)
    cv2.imshow("Segmented (diff)", seg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
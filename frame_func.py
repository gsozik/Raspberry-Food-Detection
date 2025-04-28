import cv2
import numpy as np
import json


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


def take_a_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    cv2.namedWindow("Video")

    def on_mouse(event, x, y, flags, param):
        # При нажатии левой кнопки сохраняем кадр
        if event == cv2.EVENT_LBUTTONDOWN:
            ret, frame = cap.read()
            if ret:
                filename = "frames/snapshot.png"
                cv2.imwrite(filename, frame)
                print(f"Снимок сохранён: {filename}")

    cv2.setMouseCallback("Video", on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Поток камеры завершился")
            break

        cv2.imshow("Video", frame)
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def ready_image():
    with open("seg_config.json", "r") as f:
        config = json.load(f)
    take_a_frame()
    frame = cv2.imread("frames/snapshot.png")
    back = cv2.imread("frames/background.png")
    if frame is None:
        print("Не найден файл snapshot.png")
        exit(1)
    if back is None:
        print("Не найден файл background.png")
        exit(1)

    mask, segmented = segment_by_subtraction(frame,
                                             back,
                                             config['thresh'],
                                             config['open_k'],
                                             config['close_k'])

    cv2.imwrite("frames/mask.png", mask)
    cv2.imwrite("frames/segmented.png", segmented)
    print("Сегментация завершена: mask.png, segmented.png")

    #    cv2.imshow("Mask", mask)
    #    cv2.imshow("Segmented", segmented)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
    return mask, segmented

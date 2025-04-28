import cv2
import numpy as np
import json
import os
from frame_func import segment_by_subtraction
CONFIG_PATH     = 'seg_config.json'
BACKGROUND_PATH = 'frames/background.png'
SNAPSHOT_PATH   = 'frames/snapshot.png'

def load_config():
    defaults = {'thresh': 30, 'open_k': 3, 'close_k': 7}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                conf = json.load(f)
            for k, v in defaults.items():
                conf.setdefault(k, v)
            return conf
        except Exception:
            return defaults
    return defaults

def save_config(conf):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(conf, f, indent=4)

def nothing(x):
    pass

def main():
    # 1) Загрузка фона и кадра
    background = cv2.imread(BACKGROUND_PATH)
    frame      = cv2.imread(SNAPSHOT_PATH)
    if background is None or frame is None:
        print(f"Убедитесь, что в папке есть {BACKGROUND_PATH} и {SNAPSHOT_PATH}")
        return
    if background.shape[:2] != frame.shape[:2]:
        background = cv2.resize(background, (frame.shape[1], frame.shape[0]))

    # 2) Настройка окна и трекбаров
    cfg = load_config()
    cv2.namedWindow('Calibrate', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Thresh',    'Calibrate', cfg['thresh'],   255, nothing)
    cv2.createTrackbar('OpenSize',  'Calibrate', cfg['open_k'],   50,  nothing)
    cv2.createTrackbar('CloseSize', 'Calibrate', cfg['close_k'],  50,  nothing)

    while True:
        # читаем параметры
        t = cv2.getTrackbarPos('Thresh',    'Calibrate')
        o = cv2.getTrackbarPos('OpenSize',  'Calibrate') or 1
        c = cv2.getTrackbarPos('CloseSize', 'Calibrate') or 1

        # сегментация
        mask, seg = segment_by_subtraction(frame, background, t, o, c)

        # приводим к единому размеру для hstack
        h2, w2     = frame.shape[0]//2, frame.shape[1]//2
        mask_bgr   = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_small = cv2.resize(mask_bgr,   (w2, h2))
        seg_small  = cv2.resize(seg,         (w2, h2))
        combined   = np.hstack([mask_small, seg_small])

        cv2.imshow('Calibrate', combined)

        if cv2.waitKey(50) & 0xFF == 27:  # ESC
            break

    cv2.destroyAllWindows()

    # 3) Сохранение результатов и настроек
    cv2.imwrite('frames/mask_calibrated.png', mask)
    cv2.imwrite('frames/segmented_calibrated.png', seg)
    cfg_out = {'thresh': t, 'open_k': o, 'close_k': c}
    save_config(cfg_out)
    print(f"Сохранены mask_calibrated.png, segmented_calibrated.png")
    print(f"Параметры сохранены в {CONFIG_PATH}: {cfg_out}")

if __name__ == "__main__":
    main()
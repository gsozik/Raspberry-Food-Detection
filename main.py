import cv2
import json
from pathlib import Path
from frame_func import on_mouse, segment_by_background
from find_blobs import merge_overlapping_boxes, detect_blobs


def main():
    BASE = Path(__file__).parent
    FRAMES_DIR = BASE / 'frames'
    CONFIG_PATH = BASE / 'seg_config.json'
    BG_PATH = FRAMES_DIR / 'background.png'

    FRAMES_DIR.mkdir(exist_ok=True)
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    bg = cv2.imread(str(BG_PATH))
    if bg is None:
        raise FileNotFoundError(f"Фон не найден: {BG_PATH}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру")

    cv2.namedWindow('Video')
    state = {'clicked': False}
    cv2.setMouseCallback('Video', on_mouse, state)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Video', frame)

        if state['clicked']:
            snapshot = frame.copy()
            (FRAMES_DIR / 'snapshot.png').write_bytes(cv2.imencode('.png', snapshot)[1])
            print("Snapshot saved.")

            # сегментация
            mask, segmented = segment_by_background(
                snapshot, bg,
                thresh=cfg['thresh'],
                open_k=cfg['open_k'],
                close_k=cfg['close_k'],
                merge_contours=False  # теперь не сливаем их в одну оболочку
            )
            cv2.imwrite(str(FRAMES_DIR / 'mask.png'), mask)
            cv2.imwrite(str(FRAMES_DIR / 'segmented.png'), segmented)

            # детект блобов
            raw_blobs = detect_blobs(
                mask, snapshot,
                min_area=cfg['min_area'],
                min_width=cfg.get('min_width', 20),
                min_height=cfg.get('min_height', 20)
            )

            # собираем списки bbox
            bboxes = [b['bbox'] for b in raw_blobs]
            # объединяем пересекающиеся
            merged = merge_overlapping_boxes(bboxes)

            # аннотируем
            annotated = snapshot.copy()
            for x, y, w, h in merged:
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,255,0), 2)

            cv2.imshow('Annotated', annotated)
            cv2.imwrite(str(FRAMES_DIR / 'annotated_blobs.png'), annotated)
            print(f"Annotated saved with {len(merged)} boxes.")

            # сохраняем чистые ROI из original
            for i, (x, y, w, h) in enumerate(merged, start=1):
                roi = snapshot[y:y+h, x:x+w]
                cv2.imwrite(str(FRAMES_DIR / f"blob_{i:02d}.png"), roi)
            print(f"Saved {len(merged)} clean blobs.")

            state['clicked'] = False

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
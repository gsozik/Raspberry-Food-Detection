import cv2
import json
from pathlib import Path
from ultralytics import YOLO
import frame_func
import find_blobs

# --- Paths & config ---
BASE        = Path(__file__).parent
FRAMES_DIR  = BASE / 'frames'
CFG_PATH    = BASE / 'seg_config.json'
BG_PATH     = FRAMES_DIR / 'background.png'
CLS_WEIGHTS = BASE / 'food41_cls_exp4' / 'weights' / 'best.pt'

# ensure frames directory exists
FRAMES_DIR.mkdir(exist_ok=True)

# load segmentation params
cfg = json.loads(CFG_PATH.read_text(encoding='utf-8'))

# load classification model
cls_model = YOLO(str(CLS_WEIGHTS), task='classify')

# load background
bg = cv2.imread(str(BG_PATH))
if bg is None:
    raise FileNotFoundError(f"Не найден фон: {BG_PATH}")

clicked = False
def on_mouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True

def merge_overlapping_boxes(bboxes):
    """
    Merge any overlapping (x,y,w,h) boxes into combined boxes.
    """
    rects = [[x, y, x + w, y + h] for x, y, w, h in bboxes]
    merged = []
    while rects:
        x1, y1, x2, y2 = rects.pop(0)
        changed = True
        while changed:
            changed = False
            rest = []
            for xx1, yy1, xx2, yy2 in rects:
                if not (xx2 < x1 or xx1 > x2 or yy2 < y1 or yy1 > y2):
                    x1, y1 = min(x1, xx1), min(y1, yy1)
                    x2, y2 = max(x2, xx2), max(y2, yy2)
                    changed = True
                else:
                    rest.append([xx1, yy1, xx2, yy2])
            rects = rest
        merged.append((x1, y1, x2 - x1, y2 - y1))
    return merged

# set up camera window
cap = cv2.VideoCapture(0)
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', on_mouse)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Video', frame)
    key = cv2.waitKey(1) & 0xFF

    if clicked:
        # 1) save snapshot
        snap = frame.copy()
        cv2.imwrite(str(FRAMES_DIR / 'snapshot.png'), snap)

        # 2) segment by background
        mask, segmented = frame_func.segment_by_background(
            snap, bg,
            thresh=cfg['thresh'],
            open_k=cfg['open_k'],
            close_k=cfg['close_k'],
            merge_contours=False
        )
        cv2.imwrite(str(FRAMES_DIR / 'mask.png'), mask)
        cv2.imwrite(str(FRAMES_DIR / 'segmented.png'), segmented)

        # 3) detect blobs
        raw = find_blobs.detect_blobs(
            mask, snap,
            min_area=cfg['min_area'],
            min_width=cfg.get('min_width', 20),
            min_height=cfg.get('min_height', 20)
        )
        bboxes = [r['bbox'] for r in raw]

        # 4) merge overlapping boxes
        merged = merge_overlapping_boxes(bboxes)

        # 5) annotate & classify
        ann = snap.copy()
        for (x, y, w, h) in merged:
            cv2.rectangle(ann, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = snap[y:y+h, x:x+w]
            res = cls_model.predict(source=roi, verbose=False)[0]
            if hasattr(res, 'probs') and res.probs is not None:
                cls_idx  = int(res.probs.top1)
                conf     = float(res.probs.top1conf)
                cls_name = cls_model.names[cls_idx]
            else:
                cls_name, conf = "unknown", 0.0
            label = f"{cls_name} {conf:.2f}"
            ty = y - 10 if y > 20 else y + h + 20
            cv2.putText(ann, label, (x, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # show and save annotated
        cv2.imshow('Annotated', ann)
        cv2.imwrite(str(FRAMES_DIR / 'annotated.png'), ann)

        # 6) save clean blobs
        for i, (x, y, w, h) in enumerate(merged, start=1):
            blob = snap[y:y+h, x:x+w]
            cv2.imwrite(str(FRAMES_DIR / f'blob_{i:02d}.png'), blob)

        clicked = False

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
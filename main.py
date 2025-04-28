import cv2
import frame_func
import find_blobs


frame_func.ready_image()

mask = cv2.imread('frames/mask.png', cv2.IMREAD_GRAYSCALE)
frame = cv2.imread('frames/segmented.png')
snap = cv2.imread('frames/snapshot.png')

if mask is None:
    print("Не найден файл mask.png")
    exit(1)
if frame is None:
    print("Не найден файл segmented.png")
    exit(1)

blobs = find_blobs.detect_blobs(mask, frame, min_area=1000)
print(f"Найдено {len(blobs)} блобов")
# Сохраняем полученные ROI

for i, blob in enumerate(blobs, start=1):
    cv2.imwrite(f'frames/blob_{i:02d}.png', blob['roi'])
# Сохраняем аннотированное изображение

annotated = find_blobs.annotate_blobs(snap, blobs)
cv2.imwrite('frames/annotated_blobs.png', annotated)
print("ROI сохранены как blob_##.png; аннотации — annotated_blobs.png")
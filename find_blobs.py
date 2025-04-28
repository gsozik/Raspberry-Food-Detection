import cv2
import os

def detect_blobs(mask, frame, min_area=500):

    # Найти контуры на маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi = frame[y:y+h, x:x+w]
        blobs.append({
            'contour': cnt,
            'bbox': (x, y, w, h),
            'area': area,
            'roi': roi
        })
    return blobs

def annotate_blobs(frame, blobs, color=(0, 255, 0), thickness=2):
    annotated = frame.copy()
    for blob in blobs:
        x, y, w, h = blob['bbox']
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
    return annotated


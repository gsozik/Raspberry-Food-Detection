import cv2

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
                filename = "snapshot.png"
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

if __name__ == "__main__":
    take_a_frame()
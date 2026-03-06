import cv2
import numpy as np
import os

# 1. Завантаження моделі
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
config_path = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# 2. Завантаження фото
input_name = "photo.jpg"  # назва твого вхідного файлу
image = cv2.imread(input_name)

if image is None:
    print("Помилка: Не вдалося завантажити зображення. Перевір шлях!")
else:
    (h, w) = image.shape[:2]

    # 3. Препроцесинг (blob)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # 4. Прохід через нейромережу
    net.setInput(blob)
    detections = net.forward()

    # 5. Малюємо рамки
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Малюємо прямокутник
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Додаємо текст з відсотком впевненості
            text = f"{confidence * 100:.2f}%"
            cv2.putText(image, text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # 6. ЗБЕРЕЖЕННЯ КОПІЇ
    output_name = "detected_face_output.jpg"
    cv2.imwrite(output_name, image)

    print(f"Готово! Копію збережено як: {output_name}")

    # (Опціонально) Показати результат на екрані
    cv2.imshow("Detected", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
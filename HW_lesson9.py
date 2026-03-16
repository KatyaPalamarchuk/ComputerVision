import cv2
import csv
import os
import numpy as np

net = cv2.dnn.readNetFromCaffe('data/MobileNet/mobilenet_v2_deploy.prototxt', 'data/MobileNet/mobilenet_v2.caffemodel')
classes = []
with open('data/MobileNet/synset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(' ', 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

folder_path = 'image/MobileNet'
image_files = os.listdir(folder_path)
csv_data = []
stats = {}

for filename in image_files:
    img_path = os.path.join(folder_path, filename)
    image = cv2.imread(img_path)

    if image is None:
        continue

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (224, 224)),
        1.0 / 127.5,
        (224, 224),
        (127.5, 127.5, 127.5)
    )

    net.setInput(blob)
    preds = net.forward()
    index = int(np.argmax(preds[0]))

    label = classes[index] if index < len(classes) else "unknown"
    conf = float(preds[0][index].item()) * 100

    stats[label] = stats.get(label, 0) + 1
    csv_data.append([filename, label, f"{round(conf, 2)}%", stats[label]])

    cv2.imshow("result", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

with open('report.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['Файл', 'Клас', 'Впевненість', 'Кількість збігів цього класу'])
    writer.writerows(csv_data)
import cv2

# img = cv2.imread('image/1.jpg')
# img = cv2.resize(img, (500, 300))
# cv2.imshow('image', img)
cap = cv2.VideoCapture('video/2.mp4')


while True:
    ret, frame = cap.read()
    if not ret:
        break

    resize = cv2.resize(frame, (600, 300))

    cv2.imshow('video', resize)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()


import cv2
image = cv2.imread('image/selfie.jpg')
image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 3))
cv2.rectangle(image, (275, 130), (440, 270), (0, 0, 0), 2)
cv2.putText(image, "Katya Palamarchuk", (280, 120), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.6, (0, 0, 0), 1)
cv2.imwrite('image/edited_selfie.jpg', image)

cv2.imshow("Selfie", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
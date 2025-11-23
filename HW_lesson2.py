import cv2
import numpy as np
image = cv2.imread('image/photo.jpg')
image_email = cv2.imread('image/email.jpg')
print(image.shape)
image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
print(image.shape)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges_image = cv2.Canny(gray_image, 100, 100)

print(image.shape)
image_email = cv2.resize(image_email, (image_email.shape[1] // 2, image_email.shape[0] // 2))
print(image_email.shape)
gray_email = cv2.cvtColor(image_email, cv2.COLOR_BGR2GRAY)
edges_email = cv2.Canny(gray_email, 100, 100)



cv2.imshow("Photo", image)
cv2.imshow("Gray Photo", gray_image)
cv2.imshow("Edges Photo", edges_image)
cv2.imshow("Email" ,image_email)
cv2.imshow("Gray Email" ,gray_email)
cv2.imshow("Edges Email", edges_email)
cv2.waitKey(0)

cv2.destroyAllWindows()

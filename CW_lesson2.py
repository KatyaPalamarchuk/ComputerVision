import cv2
import numpy as np
image = cv2.imread('image/1.jpg')
print(image.shape)
#image = cv2.resize(image, (800, 500))
image = cv2.resize(image, (image.shape[1] // 6, image.shape[0] // 6))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image.shape)
image = cv2.Canny(image, 100, 100)


Kernel = np.ones((5, 5), np.uint8)
image = cv2.dilate(image, Kernel, iterations=1) #диляція - розширює світлі області
image = cv2.erode(image, Kernel, iterations=1)
cv2.imwrite('1.jpg', image)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
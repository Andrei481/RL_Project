import cv2

image = cv2.imread('cat.jpg')
name = 'Andrei Joldea'
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
color = (0, 0, 255)
thickness = 3
(text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, thickness)
position = (50, image.shape[0] - 50)

cv2.putText(image, name, position, font, font_scale, color, thickness)
cv2.imshow('Image Display', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('cat_new.jpg', image)

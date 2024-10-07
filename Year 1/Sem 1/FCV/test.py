import cv2

# Load an image from disk
image = cv2.imread('cat.jpg')

# Define the text to be added (replace with your name)
name = 'Andrei Joldea'

# Set the font, size, color, and thickness of the text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
color = (0, 0, 255)  # Blue in BGR
thickness = 3

# Get the size of the text box
(text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, thickness)

# Position the text at the bottom left of the image
position = (50, image.shape[0] - 50)

# Write the name on the image
cv2.putText(image, name, position, font, font_scale, color, thickness)

# Display the image with the name written on it
cv2.imshow('Image Display', image)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the modified image to disk
cv2.imwrite('cat_new.jpg', image)

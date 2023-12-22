import cv2
import numpy as np
from google.colab.patches import cv2_imshow

def detect_stamps(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 500
    stamp_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    result_image = cv2.drawContours(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), stamp_contours, -1, (0, 255, 0), 2)

    return result_image

output_image = detect_stamps('/content/Media.jpg')
cv2_imshow(output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

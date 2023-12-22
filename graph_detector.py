import cv2
import numpy as np
from google.colab.patches import cv2_imshow

source_img = cv2.imread('/content/barimg.png', 0)

edges = cv2.Canny(source_img, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result_img = cv2.cvtColor(source_img, cv2.COLOR_GRAY2BGR)

detected_graphs = []

for contour in contours:
    if cv2.contourArea(contour) > 1000:  
      
        detected_graphs.append(contour)
        cv2.drawContours(result_img, [contour], -1, (0, 255, 0), 2)

cv2_imshow(result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Number of detected graphs: {len(detected_graphs)}")

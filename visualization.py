import numpy as np
import cv2


#===========================#
#                           #
#         Functions         #
#                           #
#===========================#

def overlap(image, points, white=True):
    copy        = image.copy()
    left        = points[0]
    top         = points[1]
    right       = points[2]
    bottom      = points[3]
    fragment    = copy[top: bottom, left: right]

    if white is True:
        eclipse = np.ones(fragment.shape, dtype=np.uint8) * 255
    else:
        eclipse = np.zeros(fragment.shape, dtype=np.uint8)

    res         = cv2.addWeighted(fragment, 0.5, eclipse, 0.5, 1.0)
    copy[top: bottom, left: right] = res
    return copy

def rectangle(image, boxes):
    
    copy = image.copy()
    
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    return copy

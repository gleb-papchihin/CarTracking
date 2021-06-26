from shapely import geometry
import numpy as np
import math
import cv2

#===========================#
#                           #
#         Classes           #
#                           #
#===========================#

class Transform:

    def __init__(self, shape, size=608, fragment=None):
        
        ''' 
            shape:      Array of image properties [width, height]

            fragment:   Array of points (left, top, right, bottom).
                        If framgment is not None, Image will be cropped.

            size:       Integer (512, 608). Required.
                        Image will be resized to square.
        '''

        self.fragment   = fragment
        self.shape      = shape
        self.size       = size

    def crop_frame(self, frame):
        
        if self.fragment == None:
            return frame

        left    = self.fragment[0]
        top     = self.fragment[1]
        right   = self.fragment[2]
        bottom  = self.fragment[3]
        cropped = frame[top: bottom, left: right]
        return cropped

    def resize_frame(self, frame):
        resized = cv2.resize(frame, (self.size, self.size))
        return resized

    def convert_frame_to_blob(self, frame, swapRB=True):
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, swapRB=swapRB, mean=(0,0,0), crop=False)
        return blob

    def transform(self, frame):
        cropped = self.crop_frame(frame)
        resized = self.resize_frame(cropped)
        return resized

    def get_blob(self, frame):
        resized = self.transform(frame)
        blob    = self.convert_frame_to_blob(resized)
        return blob

    def convert_to_origin(self, box):

        if self.fragment != None:
            width   = self.fragment[2] - self.fragment[0]
            height  = self.fragment[3] - self.fragment[1]
            scale_x = width / self.size
            scale_y = height / self.size
            shift_x = self.fragment[0]
            shift_y = self.fragment[1]
        else:
            width   = self.shape[0]
            height  = self.shape[1]
            scale_x = width / self.size
            scale_y = height / self.size
            shift_x = 0
            shift_y = 0

        left    = int(box[0] * scale_x + shift_x)
        top     = int(box[1] * scale_y + shift_y)
        right   = int(box[2] * scale_x + shift_x)
        bottom  = int(box[3] * scale_y + shift_y)

        return left, top, right, bottom

    def __call__(self, frame):
        blob = self.get_blob(frame)    
        return blob

class Detector:

    def __init__(self, weights, config, target=None):

        '''
            weights:    Path to yolo.weights

            config:     Path to yolo.cfg

            target:     Array of object id.
        '''

        self.net    = cv2.dnn.readNetFromDarknet(config, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.layers = self.getOutputsNames(self.net)
        self.target = target

    def getOutputsNames(self, net):
        layersNames = net.getLayerNames()
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def detect(self, blob):
        self.net.setInput(blob)
        detections = self.net.forward(self.layers)
        return detections

    def sort_detections(self, detections, shape, threshold):
        confidences = []
        indices     = []
        boxes       = []
        width       = shape[0]
        height      = shape[1]
        
        for detection in detections[0]:
            scores = detection[5:]
            index  = np.argmax(scores)
            conf   = scores[index]

            if self.target != None:
                if index not in self.target:
                    continue
            
            if conf > threshold:
                center_x = int(detection[0] * width)      
                center_y = int(detection[1] * height)
                w        = int(detection[2] * width)
                h        = int(detection[3] * height)
                left     = int(center_x - w / 2)
                top      = int(center_y - h / 2)
                box      = [left, top, w, h]
                
                boxes.append(box)
                indices.append(index)
                confidences.append(conf)
                
        return indices, boxes, confidences

    def remove_intersections(self, detections, conf_thr, nms_thr=0.4):
        boxes       = detections[1]
        handler     = lambda x: float(x)
        confidences = list(map(handler, detections[2]))
        indices     = cv2.dnn.NMSBoxes(boxes, confidences, conf_thr, nms_thr)

        new_boxes   = []
        new_confs   = []
        new_indices = []

        for index in indices:
            i = index[0]
            new_boxes.append(boxes[i])
            new_confs.append(confidences[i])
            new_indices.append(detections[0][i])

        return new_indices, new_boxes, new_confs

    def convert_size_to_box(self, size):
        left, top, width, height = map(lambda x: int(x), size)
        right                    = left + width
        bottom                   = top + height
        box    = [left, top, right, bottom]
        return box

    def convert_sizes_to_boxes(self, sizes):
        boxes = []
        for size in sizes:
            box = self.convert_size_to_box(size)
            boxes.append(box)
        return boxes

    def __call__(self, blob, threshold=0.6):
        height     = blob[0][0].shape[0]
        width      = blob[0][0].shape[1]
        shape      = [width, height]
        detections = self.detect(blob)
        detections = self.sort_detections(detections, shape, threshold)
        indices, boxes, confs = self.remove_intersections(detections, threshold)
        boxes      = self.convert_sizes_to_boxes(boxes)
        return indices, boxes, confs

class Trackers:

    def __init__(self):
        '''
            directions: Dict of angles (360-degree).

            previous:   Dict of previous positions [left, top, right, bottom].

            trackers:   Dict of trackers.
        '''

        self.directions  = {}
        self.previous    = {}
        self.trackers    = {}
        self.index       = 0

    def create_index(self):
        index       = self.index
        self.index  = index + 1
        
        if self.index > 1e5:
            self.index = 0
            
        return index

    def get_magnitude(self, prev_box, box):
        length      = len(prev_box)
        handler     = lambda i: abs(prev_box[i] - box[i])
        difference  = list( map( handler, range(0, length) ) )
        magnitude   = sum(difference)
        return magnitude

    def get_angle(self, prev_center, center):
        px, py = prev_center
        x, y   = center
        vect_x = x - px
        vect_y = -(y - py)
        
        if (py == y) and (px < x):
            return 0
        elif (px == x) and (py < y):
            return 90
        elif (py == y) and (px > x):
            return 180
        elif (px == x) and (py > y):
            return 270
        
        pos_vect_x = abs(vect_x)
        pos_vect_y = abs(vect_y)
        hypotenuse = pos_vect_x**2 + pos_vect_y**2
        hypotenuse = math.sqrt(hypotenuse)
        
        if hypotenuse == 0:
            return None
        
        sin        = pos_vect_y / hypotenuse
        rad        = math.asin(sin)
        angle      = self.convert_rad_to_degree(rad)
        angle      = round(angle, 1)
    
        if (vect_x > 0) and (vect_y > 0):
            return angle
        elif (vect_x < 0) and (vect_y > 0):
            return 180 - angle
        elif (vect_x < 0) and (vect_y < 0):
            return 180 + angle
        elif (vect_x > 0) and (vect_y < 0):
            return 360 - angle

    def get_area_of_intersection(self, updated, detected):
        u_rectangle  = geometry.box(*updated)
        d_rectangle  = geometry.box(*detected)
        intersection = u_rectangle.intersection(d_rectangle)
        area         = intersection.area
        
        if u_rectangle.covers(d_rectangle):
            return u_rectangle.area
        
        return area

    def get_index(self, updates, detected, threshold=0.6):
        max_match = threshold
        max_index = None
        
        for index, update in updates.items():
            box   = update[1]
            match = self.match(box, detected)
            
            if match > max_match:
                max_match = match
                max_index = index
            
        return max_index

    def convert_box_to_size(self, box):
        left, top, right, bottom = box
        width  = right - left
        height = bottom - top
        return left, top, width, height

    def convert_size_to_box(self, size):
        left, top, width, height = map(lambda x: int(x), size)
        right                    = left + width
        bottom                   = top + height
        box    = [left, top, right, bottom]
        return box

    def convert_rad_to_degree(self, rad):
        factor = math.pi / 180
        return rad / factor

    def add_tracker(self, tracker):
        index = self.create_index()
        self.trackers.update({index: tracker})
        self.directions.update({index: None})
        self.previous.update({index: None})

    def drop_tracker(self, index):
        self.directions.pop(index, None)
        self.previous.pop(index, None)
        self.trackers.pop(index, None)

    def save_to_previous(self, index, box):
        self.previous.update({index: box})

    def save_to_direction(self, index, box):
        prev_box       = self.previous[index]
        pl, pt, pr, pb = prev_box
        l, t, r, b     = box
        px_center      = ((pr - pl) / 2) + pl
        py_center      = ((pb - pt) / 2) + pt
        x_center       = ((r - l) / 2) + l
        y_center       = ((b - t) / 2) + t
        prev_center    = [px_center, py_center]
        center         = [x_center, y_center]
        angle          = self.get_angle(prev_center, center)
        self.directions.update({index: angle})

    def match(self, updated, detected):
        rectangle    = geometry.box(*updated)
        area         = rectangle.area
        intersection = self.get_area_of_intersection(updated, detected)
        area         = area if area > 0 else 1
        return intersection / area

    def update(self, frame, min_magnitude=2):
        updates = {}
        drop    = []
        
        for index, tracker in self.trackers.items():
            update    = tracker.update(frame)
            box       = self.convert_size_to_box(update[1])
            status    = update[0]
            
            if status is False:
                drop.append(index)
                continue
            
            updates.update({index: (status, box)})
            prev_box  = self.previous.get(index)
            
            if prev_box == None:
                self.save_to_previous(index, box)
                continue
            
            magnitude = self.get_magnitude(prev_box, box)
            
            if magnitude > min_magnitude:
                self.save_to_direction(index, box)
                self.save_to_previous(index, box)
        
        for index in drop:
            self.drop_tracker(index)

        return updates

class Horizon:

    def __init__(self, boundary):
        self.boundary = boundary

    def is_nested(self, box):
        left   = (box[0] >= self.boundary[0])
        top    = (box[1] >= self.boundary[1])
        right  = (box[2] <= self.boundary[2])
        bottom = (box[3] <= self.boundary[3])
        return all([left, top, right, bottom])

    def is_crossed(self, box, direction):
        top = []
        top.append(box[1] <= self.boundary[1])
        top.append(direction <= 180)

        #========#
        # TOP    #
        #========#

        if all(top):
            return True, 'top'

        left = []
        left.append(box[0] <= self.boundary[0])
        left.append(direction <= 180)

        if all(left):
            return True, 'top'

        right = []
        right.append(box[2] >= self.boundary[2])
        right.append(direction <= 180)

        if all(right):
            return True, 'top'


        #========#
        # BOTTOM #
        #========#

        left = []
        left.append(box[0] <= self.boundary[0])
        left.append(direction > 180)

        if all(left):
            return True, 'bottom'

        bottom = []
        bottom.append(box[3] >= self.boundary[3])
        bottom.append(direction > 180)

        if all(bottom):
            return True, 'bottom'

        right = []
        right.append(box[2] >= self.boundary[2])
        right.append(direction > 180)

        if all(right):
            return True, 'bottom'

        return False, None

class Vertical:

    def __init__(self, boundary):
        self.boundary = boundary

    def is_nested(self, box):
        left   = (box[0] >= self.boundary[0])
        top    = (box[1] >= self.boundary[1])
        right  = (box[2] <= self.boundary[2])
        bottom = (box[3] <= self.boundary[3])
        return all([left, top, right, bottom])

    def is_crossed(self, box, direction):
        left = []
        left.append(box[0] <= self.boundary[0])
        left.append(box[2] > self.boundary[0])
        left.append(90 <= direction <= 270)

        if all(left):
            return True, 'left'

        right = []
        right.append(box[0] < self.boundary[2])
        right.append(box[2] >= self.boundary[2])
        right.append((0 < direction < 90) or (270 < direction < 360))

        if all(right):
            return True, 'right'

        return False, None

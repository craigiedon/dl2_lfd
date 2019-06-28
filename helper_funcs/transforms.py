import cv2

class Crop(object):
    def __init__(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
    
    def __call__(self, img):
        return img[self.top:self.bottom, self.left:self.right, :]


class Resize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width
    
    def __call__(self, img):
        return cv2.resize(img, (self.height, self.width))
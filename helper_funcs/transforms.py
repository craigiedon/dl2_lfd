import cv2

class Crop(object):
    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width
    
    def __call__(self, img):
        return img[self.top:self.top + self.height,
                   self.left:self.left + self.width,
                   :]


class Resize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width
    
    def __call__(self, img):
        return cv2.resize(img, (self.height, self.width))

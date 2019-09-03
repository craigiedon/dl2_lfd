class Crop(object):
    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width
    
    def __call__(self, img):
        return img.crop((self.left, self.top, self.left + self.width, self.top + self.height))
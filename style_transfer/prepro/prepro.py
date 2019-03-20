
import cv2

def imread(fp):
    """
    Handle file path exceptions and return numpy array, include a return mode
    """
    # TODO: implement exceptions
    return cv2.imread(fp)

def imsave(fp, img):
    """
    Handle file path exceptions and return numpy array, include a return mode
    """
    # TODO: implement exceptions
    cv2.imwrite(fp, img)

import numpy as np
import cv2
import base64

def readb64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
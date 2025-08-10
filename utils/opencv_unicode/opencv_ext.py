import os

import cv2
import numpy as np


def _is_ascii(path):
    try:
        path.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def imread(filename, flags=cv2.IMREAD_COLOR):
    if _is_ascii(filename):
        return cv2.imread(filename, flags)
    data = np.fromfile(filename, dtype=np.uint8)
    return cv2.imdecode(data, flags)


def imwrite(filename, img, params=None):
    ext = os.path.splitext(filename)[1]
    if _is_ascii(filename):
        return cv2.imwrite(filename, img, params)
    success, buf = cv2.imencode(ext, img, params or [])
    if success:
        buf.tofile(filename)
    return success

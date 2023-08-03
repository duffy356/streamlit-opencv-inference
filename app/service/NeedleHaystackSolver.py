import collections

import numpy as np
import cv2

LocatedResult = collections.namedtuple('LocatedResult', 'confidence box')
Box = collections.namedtuple('Box', 'left top width height')

class NeedleHaystackSolver:

    @staticmethod
    def locate_all_opencv(needleImage, haystackImage, grayscale=True, limit=10_000, region=None, step=1,
                          confidence=0.999):
        confidence = float(confidence)

        needleImage = NeedleHaystackSolver.load_cv2(needleImage, grayscale)
        needleHeight, needleWidth = needleImage.shape[:2]
        haystackImage = NeedleHaystackSolver.load_cv2(haystackImage, grayscale)

        if region:
            haystackImage = haystackImage[region[1]:region[1]+region[3],
                            region[0]:region[0]+region[2]]
        else:
            region = (0, 0)  # full image; these values used in the yield statement
        if (haystackImage.shape[0] < needleImage.shape[0] or
                haystackImage.shape[1] < needleImage.shape[1]):
            # avoid semi-cryptic OpenCV error below if bad size
            raise ValueError('needle dimension(s) exceed the haystack image or region dimensions')

        if step == 2:
            confidence *= 0.95
            needleImage = needleImage[::step, ::step]
            haystackImage = haystackImage[::step, ::step]
        else:
            step = 1

        # get all matches at once, credit: https://stackoverflow.com/questions/7670112/finding-a-subimage-inside-a-numpy-image/9253805#9253805
        result = cv2.matchTemplate(haystackImage, needleImage, cv2.TM_CCOEFF_NORMED)
        match_indices = np.arange(result.size)[(result > confidence).flatten()]
        matches = np.unravel_index(match_indices[:limit], result.shape)

        if len(matches[0]) == 0:
            return None

        # use a generator for API consistency:
        matchx = matches[1] * step + region[0]  # vectorized
        matchy = matches[0] * step + region[1]
        for x, y in zip(matchx, matchy):
            result_conf = result[y][x]
            yield LocatedResult(result_conf, Box(x, y, needleWidth, needleHeight))

    @staticmethod
    def load_cv2(img, grayscale=None):
        if isinstance(img, (str, str)):
            # The function imread loads an image from the specified file and
            # returns it. If the image cannot be read (because of missing
            # file, improper permissions, unsupported or invalid format),
            # the function returns an empty matrix
            # http://docs.opencv.org/3.0-beta/modules/imgcodecs/doc/reading_and_writing_images.html
            if grayscale:
                img_cv = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            else:
                img_cv = cv2.imread(img, cv2.IMREAD_COLOR)
            if img_cv is None:
                raise IOError("Failed to read %s because file is missing, "
                              "has improper permissions, or is an "
                              "unsupported or invalid format" % img)
        elif isinstance(img, np.ndarray):
            # don't try to convert an already-gray image to gray
            if grayscale and len(img.shape) == 3:  # and img.shape[2] == 3:
                img_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_cv = img
        elif hasattr(img, 'convert'):
            # assume its a PIL.Image, convert to cv format
            img_array = np.array(img.convert('RGB'))
            img_cv = img_array[:, :, ::-1].copy()  # -1 does RGB -> BGR
            if grayscale:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            raise TypeError('expected an image filename, OpenCV numpy array, or PIL image')
        return img_cv

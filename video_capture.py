import cv2 as cv
from tqdm import tqdm


def iter_video(path, verbose=True):
    capture = cv.VideoCapture(path)
    nframes = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    iterator = tqdm(range(nframes)) if verbose else range(nframes)
    for _ in iterator:
        success, bgr = capture.read()
        assert success
        assert len(bgr.shape) == 3  # height, width, depth
        assert bgr.shape[2] == 3  # blue, green, red
        yield bgr[:, :, ::-1]

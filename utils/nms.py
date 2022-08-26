from .iou import compute_iou
import numpy as np


def nms(dets: np.ndarray, iou_thresh: float = 0.4):
    """ Non maximum suppression

    :param dets:        array(N, 5), [xmin, ymin, xmax, ymax, score]
    :param iou_thresh:  float, thresh of IoU
    :return:            bbox indices which should be keep
    """
    scores = dets[:, 4]

    indices = np.argsort(scores)[::-1]
    ans = []
    while len(indices):
        top_idx = indices[0]
        ans.append(top_idx)
        indices = indices[1:]

        # compute iou
        ious = compute_iou(bb1=dets[indices, None],
                           bb2=np.repeat(dets[top_idx, None], repeats=len(indices), axis=0))

        # suppression
        indices = indices[ious < iou_thresh]

    return ans

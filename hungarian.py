import numpy as np
from scipy.spatial.distance import cdist, euclidean
from scipy.optimize import linear_sum_assignment as lsa

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {0, 2, 1, 3}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 1, 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 100

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    
    return -iou #+ dist

def distance_matrix(trackers, boxes):
    """
    Get distance matrix between tracker boxes and detection boxes
    using scipy's cdist function and get_iou as the metric
    Parameters:
    ----------
    trackers: `numpy.ndarray`
        boxes of trackers of the form N x (x1, y1, x2, y2)
    boxes: `numpy.ndarray`
        detection boxes of the form M x (x1, y1, x2, y2)

    Returns:
    --------
    distance_matrix: `numpy.ndarray`
        N x M array with distances
    """
    assert len(trackers.shape) == 2
    assert len(boxes.shape) == 2 
    # trackers = format_values(trackers)  # Load trackers
    # boxes = format_values(boxes)        # Load centroids of new box predictions
    distance_matrix = cdist(trackers, boxes, get_iou)

    return distance_matrix

def get_assignment_matrix(distance_matrix):
    """
    Get hungarian assignment using scipy's linear_sum_assignment
    Parameters:
    ----------
    distance_matrix: `numpy.ndarray`
        N x M array with distances of N trackers and M detections

    Returns:
    --------
    rows: `numpy.ndarray`
        sorted list of indices of trackers (starting with 0,1,2...)
    cols: `numpy.ndarray`
        array of indices of respective assigned boxes to the trackers
    """
    rows, cols = lsa(distance_matrix)
    return rows, cols

def assigned(t_indices, b_indices, rows, cols, distances):
    """
    Get assigned and unassingned trackers and assigned & unassigned boxes

    Parameters:
    -----------
    t_indices: `numpy.ndarray`
        numpy.arange of len(trackers) to get all indices of trackers
    b_indices: `numpy.ndarray`
        numpy.arange of len(boxes) to get all indices of boxes
    rows: `numpy.ndarray`
        sorted list of indices of trackers (starting with 0,1,2...)
    cols: `numpy.ndarray`
        array of indices of respective assigned boxes to the trackers
    distances: `numpy.ndarray`
        N x M array with distances

    Returns:
    --------
    a_track: `numpy.ndarray`
        indices of trackers with assigned boxes
    a_box:
        indices of boxes assigned to trackers
    u_track:
        indices of trackers that were not assigned any boxes
    u_box:
        indices of boxes that were not assigned to any tracker
    """
    a_track = []
    u_track = []
    a_box = []
    u_box = []
    
    for r, c in zip(rows, cols):
        if distances[r, c].sum() <= -0.6:
            a_track.append(r)
            a_box.append(c)
    
    u_track = np.setdiff1d(t_indices, a_track)
    u_box = np.setdiff1d(b_indices, a_box)

    return a_track, a_box, u_track, u_box
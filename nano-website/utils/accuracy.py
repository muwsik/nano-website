import numpy as np

def qualityEstimation(blobs_gt, blobs_est, thres = 0.25):
    # [x, y, d] -> [y, x, r]
    temp_blobs_gt = blobs_gt[:, [1, 0, 2]].copy()
    temp_blobs_est = blobs_est[:, [1, 0, 2]].copy()

    temp_blobs_gt[:, 2] /= 2
    temp_blobs_est[:, 2] /= 2

    # ROI is determined only by GT blobs
    y_min = np.min(temp_blobs_gt[:, 0] - temp_blobs_gt[:, 2])
    x_min = np.min(temp_blobs_gt[:, 1] - temp_blobs_gt[:, 2])
    y_max = np.max(temp_blobs_gt[:, 0] + temp_blobs_gt[:, 2])
    x_max = np.max(temp_blobs_gt[:, 1] + temp_blobs_gt[:, 2])

    roi = np.array([
        y_min,
        x_min,
        y_max - y_min,
        x_max - x_min
    ])

    # Keep indices of blobs relative to the original arrays
    temp_blobs_gt, gt_roi_indexes = blobs_in_roi(
        temp_blobs_gt,
        roi
    )
    temp_blobs_est, est_roi_indexes = blobs_in_roi(
        temp_blobs_est,
        roi
    )

    length_gt = temp_blobs_gt.shape[0]
    length_est = temp_blobs_est.shape[0]

    # IoU matrix
    iou = np.zeros((length_gt, length_est))
    for i in range(length_gt):
        for j in range(length_est):
            iou[i, j] = findIOU4circle(
                temp_blobs_gt[i],
                temp_blobs_est[j]
            )

    # Matching matrix
    match_matr = np.zeros(
        (length_gt, length_est),
        dtype = int
    )

    for i in range(length_gt):
        if length_est > 0 and np.max(iou[i]) >= thres:
            imax = np.argmax(iou[i])
            match_matr[i, imax] = 1

    # Resolve cases where several GT blobs
    # are matched to the same estimated blob
    fake_index = np.zeros(length_est, dtype = bool)
    truedetected_blobs_index = np.zeros(
        length_est,
        dtype = bool
    )

    for j in range(length_est):
        if np.sum(match_matr[:, j]) > 1:
            imax = np.argmax(iou[:, j])
            match_matr[:, j] = 0
            match_matr[imax, j] = 1

        if np.sum(match_matr[:, j]) == 0:
            fake_index[j] = True
        else:
            truedetected_blobs_index[j] = True

    # Determine FN / TP
    no_match_index = np.zeros(length_gt, dtype = bool)
    match_index = np.zeros(length_gt, dtype = bool)
    for i in range(length_gt):
        if np.sum(match_matr[i, :]) == 0:
            no_match_index[i] = True
        elif np.sum(match_matr[i, :]) == 1:
            match_index[i] = True
        else:
            raise RuntimeError("GT blob has more than one match.")

    # Convert ROI-relative indices back to indices
    # of the original blobs_gt / blobs_est arrays    
    # Return original blobs in [x, y, d] format
    FN = blobs_gt[np.flatnonzero(gt_roi_indexes)[no_match_index]]
    FP = blobs_est[np.flatnonzero(est_roi_indexes)[fake_index]]
    TP = blobs_gt[np.flatnonzero(gt_roi_indexes)[match_index]]
    # True Detected estimated blobs
    TD = blobs_est[np.flatnonzero(est_roi_indexes)[truedetected_blobs_index]]

    TD_iou = np.array([
        iou[np.where(match_matr[:, j] == 1)[0][0], j]
        for j in np.flatnonzero(truedetected_blobs_index)
    ])

    return FN, FP, TP, TD, TD_iou 


def blobs2roi(_blobs, _heightImg, _widthImg):
    roi = np.zeros(4, dtype='int')
    roi[0] = max(0, (_blobs[:,0]-_blobs[:,2]).min()) 
    roi[1] = max(0, (_blobs[:,1]-_blobs[:,2]).min())
    roi[2] = min(_heightImg, (_blobs[:,0]+_blobs[:,2]).max() - roi[0])
    roi[3] = min(_widthImg, (_blobs[:,1]+_blobs[:,2]).max() - roi[1])
    return roi 


def blobs_in_roi(blobs, roi):
    """Check if the center of blob is inside ROI  
    
    Arguments
    blobs -- list or array of areas occupied by the nanoparticle 
            (y, x, r) y and x are coordinates of the center and r - radius    
    roi -- (y,x,h,w)
    
    Return blobs list
    """
    indexes = list(map(lambda blob: int(blob[0]) >= roi[0] \
                                and int(blob[1]) >= roi[1] \
                                and int(blob[0]) < roi[0]+roi[2]  \
                                and int(blob[1]) < roi[1]+roi[3], \
                                    blobs))
    return np.copy(blobs[indexes]), indexes
    
def findIOU4circle(c1, c2):
    """Finds Jaccard similarity measure for two circles, 
       defined by the coordinates of centers and radii.
       c1=[x1,y1,r1], c2=[x2,y2,r2]  
    """

    d = np.linalg.norm(c1[:2] - c2[:2]) #distance betweem centers
        
    rad1sqr = c1[2] ** 2
    rad2sqr = c2[2] ** 2

    if d == 0:
        # the circle centers are the same
        return min(rad1sqr, rad2sqr)/max(rad1sqr, rad2sqr)

    angle1 = (rad1sqr + d ** 2 - rad2sqr) / (2 * c1[2] * d)
    angle2 = (rad2sqr + d ** 2 - rad1sqr) / (2 * c2[2] * d)

    # check if the circles are overlapping
    if (-1 <= angle1 < 1) or (-1 <= angle2 < 1):
        theta1 = np.arccos(angle1) * 2
        theta2 = np.arccos(angle2) * 2

        area1 = (0.5 * theta2 * rad2sqr) - (0.5 * rad2sqr * np.sin(theta2))
        area2 = (0.5 * theta1 * rad1sqr) - (0.5 * rad1sqr * np.sin(theta1))

        return (area1 + area2)/(np.pi*(rad1sqr+rad2sqr) - area1 - area2)

    elif angle1 < -1 or angle2 < -1:
        # Smaller circle is completely inside the largest circle.
        # Intersection area will be area of smaller circle
        # return area(c1_r), area(c2_r)
        return min(rad1sqr, rad2sqr)/max(rad1sqr, rad2sqr)
    return 0
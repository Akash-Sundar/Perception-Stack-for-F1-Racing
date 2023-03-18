import cv2
import numpy as np
from lane import find_lane

def find_focal_length():
    import glob
    CHECKERBOARD = (8,6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 
    
    
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)*25
    prev_img_shape = None


    # Extracting path of individual image stored in a given directory
    images = glob.glob('./calibration/*.png')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (5,5),(-1,-1), criteria)
            
            imgpoints.append(corners2)
    
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
    
    
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    focal_length = (K[0,0] + K[1,1])/2
    
    return focal_length

def object_detection(image):
    # detect objects and return bb

    def IoU(a, b):
        # referring to IoU algorithm in slides
        inter_w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        inter_h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter_ab = inter_w * inter_h
        area_a = (a[3] - a[1]) * (a[2] - a[0])
        area_b = (b[3] - b[1]) * (b[2] - b[0])
        union_ab = area_a + area_b - inter_ab
        return inter_ab / union_ab

    def bbox_convert(c_x, c_y, w, h):
        return [c_x - w/2, c_y - h/2, c_x + w/2, c_y + h/2]

    def bbox_convert_r(x_l, y_l, x_r, y_r):
        return [x_l/2 + x_r/2, y_l/2 + y_r/2, x_r - x_l, y_r - y_l]

    def label_to_box_xyxy(result, threshold = 0.9):
        def grid_cell(cell_indx, cell_indy):
            stride_0 = anchor_size[1]
            stride_1 = anchor_size[0]
            return np.array([cell_indx * stride_0, cell_indy * stride_1, cell_indx * stride_0 + stride_0, cell_indy * stride_1 + stride_1])

        final_dim = [5, 10]
        input_dim = [180, 320]
        anchor_size = [(input_dim[0] / final_dim[0]), (input_dim[1] / final_dim[1])]
        validation_result = []
        result_prob = []
        for ind_row in range(final_dim[0]):
            for ind_col in range(final_dim[1]):
                grid_info = grid_cell(ind_col, ind_row)
                validation_result_cell = []
                if result[0, ind_row, ind_col] >= threshold:
                    c_x = grid_info[0] + anchor_size[1]/2 + result[1, ind_row, ind_col]
                    c_y = grid_info[1] + anchor_size[0]/2 + result[2, ind_row, ind_col]
                    w = result[3, ind_row, ind_col] * input_dim[1]
                    h = result[4, ind_row, ind_col] * input_dim[0]
                    x1, y1, x2, y2 = bbox_convert(c_x, c_y, w, h)
                    x1 = np.clip(x1, 0, input_dim[1])
                    x2 = np.clip(x2, 0, input_dim[1])
                    y1 = np.clip(y1, 0, input_dim[0])
                    y2 = np.clip(y2, 0, input_dim[0])
                    validation_result_cell.append(x1)
                    validation_result_cell.append(y1)
                    validation_result_cell.append(x2)
                    validation_result_cell.append(y2)
                    result_prob.append(result[0, ind_row, ind_col])
                    validation_result.append(validation_result_cell)
        validation_result = np.array(validation_result)
        result_prob = np.array(result_prob)
        return validation_result, result_prob

    def voting_suppression(result_box, iou_threshold = 0.5):
        votes = np.zeros(result_box.shape[0])
        for ind, box in enumerate(result_box):
            for box_validation in result_box:
                if IoU(box_validation, box) > iou_threshold:
                    votes[ind] += 1
        return (-votes).argsort()

    import onnx
    model = onnx.load('model.onnx')

    import onnxruntime as ort
    ort_sess = ort.InferenceSession('model.onnx')
    result = ort_sess.run(None, {'input': image})

    # result = model(image)
    result = result.detach().cpu().numpy()

    voting_iou_threshold = 0.5
    confi_threshold = 0.4

    bboxs, result_prob = label_to_box_xyxy(result[0], confi_threshold)
    vote_rank = voting_suppression(bboxs, voting_iou_threshold)
    bbox = bboxs[vote_rank[0]]
    [c_x, c_y, w, h] = bbox_convert_r(bbox[0], bbox[1], bbox[2], bbox[3])
    bounding_box = np.array([[c_x, c_y, w, h]])

    # bounding_box = np.ones((4,2))
    return bounding_box

def lane_detection(image):
    find_lane(image)

def find_distance(KNOWN_WIDTH, focal_length, bb_mean):
    distance = KNOWN_WIDTH*focal_length / bb_mean
    return distance

def plot_bounding_box(image, bounding_box, distance):
    box = bounding_box
    box = np.int0(box)
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    cv2.putText(image, "%.2fcm" % (distance),
        (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 255, 0), 3)
    cv2.imshow("image - 40", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    focal_length = find_focal_length()

    image = cv2.imread("resource/test_car_x60cm.png")
    KNOWN_DISTANCE = 60
    bounding_box = object_detection(image)
    bb_centre = np.mean(bounding_box)
    distance = find_distance(40, focal_length, bb_centre)
    focal_length = focal_length*KNOWN_DISTANCE/distance
    distance = find_distance(40, focal_length, bb_centre)

    plot_bounding_box(image, bounding_box, distance)

    image = cv2.imread("resource/test_car_unknown.png")
    bounding_box = object_detection(image)
    bb_centre = np.mean(bounding_box)
    distance = find_distance(40, focal_length, bb_centre)

    plot_bounding_box(image, bounding_box, distance)

    image = cv2.imread("resource/lane.png")
    lane_detection(image)



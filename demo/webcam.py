# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

import numpy as np
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time
_col = (
    (0,255,0),
    (255,0,0)
)

def draw_flow(img, flow, flow_clusters=None, step=16, K=2):
    """
    Draw optical flow vectors
    """
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    #vis = img.copy()
    #vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if flow_clusters is not None:
        l = flow_clusters[y,x].T
        for k in range(K):
            _x = x[l==k]
            _y = y[l==k]
            _fx = fx[l==k]
            _fy = fy[l==k]

            lines = np.vstack([_x, _y, _x+_fx, _y+_fy]).T.reshape(-1, 2, 2)
            lines = np.int32(lines + 0.5)

            cv2.polylines(img, lines, 0, _col[k])
            for (x1, y1), (_x2, _y2) in lines:
                cv2.circle(img, (x1, y1), 1, _col[k], -1)
    else:
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        cv2.polylines(img, lines, 0, (0, 255, 0))
        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)
    return img

def warp_flow(img, flow):
    """
    Warp previous flow according to the last update
    """
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def cluster_flow(flow, K=2, threshold=1): # 
    """
    Simple k-means clustering of the flow
    """
    # flow is []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    _, labels, centers = cv2.kmeans(flow.reshape(-1,2), K , None, criteria,10,flags)

    # sort clusters, we are interested in the cluster further away from  0
    _centers=(centers*centers).sum(axis=1)
    s = np.argsort(_centers)

    if _centers[s[-1]]>threshold*threshold:
        rs = np.arange(K,dtype=np.uint8)[s] # inverse index
        motion_detected = True
    else: # it's not above 0 :(
        rs = np.zeros(K,dtype=np.uint8)
        motion_detected = False

    return rs[labels.reshape(flow.shape[:2])],motion_detected


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    cam = cv2.VideoCapture(0)
    ret_val, img = cam.read()
    prevgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flow = None
    use_temporal_propagation = True
    use_spatial_propagation = True

    inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    inst.setUseSpatialPropagation(use_spatial_propagation)

    while True:
        start_time = time.time()
        ret_val, img = cam.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        if flow is not None and use_temporal_propagation:
            #warp previous flow to get an initial approximation for the current flow:
            flow = inst.calc(prevgray, gray, warp_flow(flow, flow))
        else:
            flow = inst.calc(prevgray, gray, None)
        
        flow_clusters, motion_detected = cluster_flow(flow, K=2)

        prevgray = gray
        
        if motion_detected:
            contours, hierarchy = cv2.findContours(flow_clusters, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #print(contours)
            motion_bbox = cv2.boundingRect(contours[0])
            #print(motion_bbox)

            composite = coco_demo.run_on_opencv_image(img)

            composite = cv2.rectangle(composite, motion_bbox, (0,0,255), 1 )
            #composite = cv2.drawContours(composite, contours, -1, (0,0,255), 3)
            #composite = draw_flow(composite, flow, flow_clusters=flow_clusters)

        else:
            composite = img # don't detect if there is no motion

        #print("Time: {:.2f} s / img".format(time.time() - start_time))
        cv2.imshow("COCO detections", composite)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
        

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

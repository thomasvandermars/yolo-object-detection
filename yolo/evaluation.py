import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from .post_processing import decode, non_max_suppression

def iou(pred, true):
    """
    Function to compute Intersection over Union (IoU) of predicted boxes 
    with the corresponding ground truth boxes.
    
    :param tf.Tensor pred: predicted bounding box [M, 4].
    :param tf.Tensor true: ground truth bounding box  [N, 4].
    
    :return tf.Tensor iou: Intersection over Union values.
    """
    
    # Shapes of incoming parameters
    # pred_... --> [M, 4]
    # true_... --> [N, 4]
    
    # returns --> [M, N]
    
    pred = tf.cast(tf.expand_dims(pred, axis = 1), tf.float32) # [M, 1, 4]
    true = tf.cast(tf.expand_dims(true, axis = 0), tf.float32) # [1, N, 4]
    
    # order of predicted proposals is [xmin, ymin, xmax, ymax]
    pred_y_min = pred[...,0] # [M, 1]
    pred_x_min = pred[...,1] # [M, 1]
    pred_y_max = pred[...,2] # [M, 1]
    pred_x_max = pred[...,3] # [M, 1]
    
    true_y_min = true[...,0] # [1, N]
    true_x_min = true[...,1] # [1, N]
    true_y_max = true[...,2] # [1, N]
    true_x_max = true[...,3] # [1, N]
    
    intersect_x_min = tf.maximum(pred_x_min, true_x_min) # [M, N]
    intersect_y_min = tf.maximum(pred_y_min, true_y_min) # [M, N]
    intersect_x_max = tf.minimum(pred_x_max, true_x_max) # [M, N]
    intersect_y_max = tf.minimum(pred_y_max, true_y_max) # [M, N]
    
    intersect_width = tf.maximum(tf.subtract(intersect_x_max, intersect_x_min), 0.0) # [M, N]
    intersect_height = tf.maximum(tf.subtract(intersect_y_max, intersect_y_min), 0.0) # [M, N]
    
    intersect_area = tf.multiply(intersect_width, intersect_height) # [M, N]
    
    pred_width = tf.maximum(tf.subtract(pred_x_max, pred_x_min), 0.0) # [M, N]
    pred_height = tf.maximum(tf.subtract(pred_y_max, pred_y_min), 0.0) # [M, N]
    true_width = tf.maximum(tf.subtract(true_x_max, true_x_min), 0.0) # [M, N]
    true_height = tf.maximum(tf.subtract(true_y_max, true_y_min), 0.0) # [M, N]
    
    pred_area = tf.multiply(pred_width, pred_height) # [M, N]
    true_area = tf.multiply(true_width, true_height) # [M, N]
    
    union_area = tf.subtract(tf.add(pred_area, true_area), intersect_area) # [M, N]
    
    # we need a small number for numeric stability for when the union area happens to be 0
    iou = tf.divide(intersect_area, tf.maximum(union_area, 1e-10))
    
    return iou

def mAP(y, y_hat, conf_thres, nms_thres, iou_thres, params):
    """
    Function to calculate mean Average Precision.
    
    :param numpy.array y: Encoded target labels. Shape = (batch, GRID_H, GRID_W, classes + 5)
    :param numpy.array y_hat: Encoded predicted labels. Shape = (batch, GRID_H, GRID_W, classes + anchors * 5).
    :param float conf_thres: Object Confidence threshold. Bounding boxes with object confidence lower than 
                             this threshold are filtered out.
    :param float nms_thres: Non-Max Suppression (NMS) IoU threshold. Lower confidence boxes that exceed this 
                            IoU threshold with the higher confidence boxes are suppressed.
    :param float iou_thres: Intersection over Union (IoU) threshold. The bounding boxes remaining after 
                            non_max_suppression are assigned (as a True Positive) to a ground truth object 
                            when the predicted bounding box has an IoU with the ground truth that exceeds this 
                            threshold.
    :param dict params: Dictionary with hyperparameters.
    
    :return dict AP_by_class: Average Precision by Class
    :return float mAP: mean Average Precision
    """
    assert(isinstance(params, dict))

    CLASSES = params['CLASSES']
    
    # initiate empty lists by class for all detections made by the model
    detections = {}
    for i in CLASSES:
        detections[i] = []
        pass
    
    # array to keep track of ground truth target counts
    class_target_counts = np.zeros(len(params['CLASSES']))
    
    # iterate through labels
    for yi in range(y.shape[0]):
        
        # decode ground truth label
        gt_boxes, gt_box_conf, gt_box_class, gt_indices = decode(x = y[yi], params = params, conf_thres = 1.0)
        
        # decode predicted label
        conf_boxes, conf_scores, box_class, indices = decode(x = y_hat[yi], params = params, conf_thres = conf_thres)
    
        # apply non-max suppression to the predicted boxes (boxes returned are in descending order of confidence)
        nms_boxes, nms_conf_scores, nms_box_cls = non_max_suppression(conf_boxes, conf_scores, box_class, nms_thres)
        
        # iterate through ground truth boxes
        for i in range(len(gt_box_class)):
            # update target counts for each class
            class_target_counts[np.argmax(gt_box_class[i])] += 1
            pass
        
        # establish pred-gt IoU grid
        ious = iou(pred = nms_boxes, true = gt_boxes)
        
        # mask for keeping track of which pred-gt combinations are assigned
        assigned = np.zeros_like(ious)
        
        # FIND TRUE POSITIVES
        
        # iterate along the predicted boxes
        for i in range(ious.shape[0]):
            # iterate along the gt boxes
            for j in range(ious.shape[1]):
                
                # if the pred-gt combi exceeds the IoU threshold & is the highest IoU combi for that particular pred box
                # and we have not assigned the pred-box to any of the gt-boxes
                if (ious[i,j] > iou_thres) and (ious[i,j] == np.max(ious[i,:])) and np.sum(assigned[i,:]) == 0:
                    
                    # update assignment mask
                    assigned[i,j] = 1

                    # register TRUE POSITIVE
                    detections[CLASSES[np.argmax(nms_box_cls[i])]].append({'objectness': nms_conf_scores[i], 
                                                                           'class': CLASSES[np.argmax(nms_box_cls[i])], 
                                                                           'TP': 1.0})
                    
                    break # after the assignment we do not have to evaluate further down the gt-box axis
                    
                    pass
                pass
            
            # if all the gt-boxes are assigned, we can stop looking for True Positives
            if np.sum(assigned) == len(gt_box_conf):
                break
                
                pass
            pass
        
        # FIND FALSE POSITIVES
        
        # iterate along the predicted boxes
        for i in range(ious.shape[0]):
            
            # if the pred-box was not assigned to any gt-box...
            if np.sum(assigned[i,:]) == 0:
                
                # register FALSE POSITIVE
                detections[CLASSES[np.argmax(nms_box_cls[i])]].append({'objectness': nms_conf_scores[i], 
                                                                       'class': CLASSES[np.argmax(nms_box_cls[i])], 
                                                                       'TP': 0.0})
                pass
            pass
    
    # create a dictionary with the Average Precision information in a dataframe format
    AP_by_class = {}
    for c in CLASSES:
        sorted_detections = sorted(detections[str(c)], key=lambda d: d['objectness'], reverse=True)       
        df = pd.DataFrame.from_dict(sorted_detections)
        df['FP'] = 1.0 - df['TP']
        df['cumTP'] = df['TP'].cumsum()
        df['cumFP'] = df['FP'].cumsum()
        df['all_detections'] = df['cumTP'] + df['cumFP']
        df['Precision'] = df['cumTP'] / df['all_detections']
        df['Recall'] = df['cumTP'] / class_target_counts[CLASSES.index(c)]
        df['Recall_Range'] = df['Recall'].diff()
        df.at[0, 'Recall_Range'] = df.at[0, 'Recall']
        df['Interpolation'] = df['TP'] * df['Precision'] * df['Recall_Range']
        AP_by_class[c] = df
        pass
    
    # calculate the mean Average Precision
    APs = []
    for c in CLASSES:
        APs.append(AP_by_class[str(c)]['Interpolation'].sum())
    mAP = np.array(APs).mean()
    
    return AP_by_class, mAP

# construct a mean Average Precision curve
def mAP_curve(y, y_hat, conf_thres, nms_thres, iou_thresholds, params):
    """
    Function to create mean Average Precision curve.
    
    :param numpy.array y: Encoded target labels. Shape = (batch, GRID_H, GRID_W, classes + 5)
    :param numpy.array y_hat: Encoded predicted labels. Shape = (batch, GRID_H, GRID_W, classes + anchors * 5).
    :param float conf_thres: Object Confidence threshold. Bounding boxes with object confidence lower than 
                             this threshold are filtered out.
    :param float nms_thres: Non-Max Suppression (NMS) IoU threshold. Lower confidence boxes that exceed this 
                            IoU threshold with the higher confidence boxes are suppressed.
    :param list/1D numpy.array iou_thresholds: Intersection over Union (IoU) thresholds for which to calculate 
                                               the mean Average Precision (mAP).
    :param dict params: Dictionary with hyperparameters.
    
    :return numpy.array mAPs: Mean Average Precisions given a range of IoU thresholds.
    """
    assert(isinstance(params, dict))

    # list to hold the mean Average Precisions (mAP)
    mAPs = []
    
    # iterate through IoU thresholds
    for i in range(len(iou_thresholds)):
        # for the given IoU threshold, compute the mAP
        mAPs.append(mAP(y = y, 
                        y_hat = y_hat, 
                        conf_thres = conf_thres, 
                        nms_thres = nms_thres, 
                        iou_thres = iou_thresholds[i], 
                        params = params)[1])
        pass
    
    return np.array(mAPs)

def precision_recall_curve(AP_by_class, class_labels):
    """
    Function to show the Precision-Recall curves for the given class labels.
    
    :param dict AP_by_class: Average Precision by Class.
    :param list class_labels: class labels for which you want to show the Precision-Recall curve.
    
    :return: None
    """
    
    # iterate over class labels to be shown
    for cl in class_labels:
        # plot Precision-Recall curve
        plt.plot(AP_by_class[str(cl)]['Recall'], AP_by_class[str(cl)]['Precision'])
        pass
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve(s)')
    plt.grid(linestyle = '--')
    plt.legend(class_labels)
    pass
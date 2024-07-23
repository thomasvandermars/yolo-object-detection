import numpy as np
import tensorflow as tf

def decode(x, params, conf_thres = 1.0):
    """
    Function to filter out the bounding boxes that exceed the object confidence threshold.

    :param numpy.array x: Encoded predicted labels. Shape = (GRID_H, GRID_W, classes + anchors * 5).
    :param dict params: Dictionary with hyperparameters.
    :param float conf_thres: Object Confidence threshold. Bounding boxes with object confidence lower than 
                             this threshold are filtered out.
    
    :return numpy.array conf_boxes: bounding boxes that exceed the object confidence threshold [ymin, xmin, ymax, xmax].
    :return numpy.array box_conf: bounding box confidence scores.
    :return numpy.array box_class: bounding box class scores.
    """
    assert(isinstance(params, dict))
    
    cl = len(params['CLASSES'])
    
    # extract bounding boxes that exceed the confidence threshold from the YOLO encoding
    indices = np.array(np.where(x[:,:,:,cl] >= conf_thres)).T
    boxes = tf.gather_nd(x, indices).numpy()
    
    # extract class, object confidence, and coordinates
    box_class = tf.nn.softmax(boxes[:,:cl], axis = -1).numpy()
    box_conf = boxes[:,cl]
    box_coor = boxes[:,cl+1:]
    
    # scale up coordinates from grid-relative to (resized) image pixels
    height = box_coor[:,-1] * (params['IMG_H'] / params['GRID_H'])
    width  = box_coor[:,-2] * (params['IMG_W'] / params['GRID_W'])
    ymid   = box_coor[:,-3] * (params['IMG_H'] / params['GRID_H'])
    xmid   = box_coor[:,-4] * (params['IMG_W'] / params['GRID_W'])
    
    # convert to min-max format
    xmin = np.expand_dims((xmid - (width / 2)), axis = -1) 
    ymin = np.expand_dims((ymid - (height / 2)), axis = -1) 
    xmax = np.expand_dims((xmid + (width / 2)), axis = -1) 
    ymax = np.expand_dims((ymid + (height / 2)), axis = -1) 
    
    conf_boxes = np.concatenate([ymin, xmin, ymax, xmax], axis = -1)
    
    return conf_boxes, box_conf, box_class, indices

def non_max_suppression(conf_boxes, conf_scores, box_class, nms_thres):
    """
    Function to apply non_max_suppression to the bounding boxes that exceed the confidence threshold.
    
    :param numpy.array conf_boxes: bounding boxes that exceed the object confidence threshold [ymin, xmin, ymax, xmax]. Shape = [num_boxes, 4].
    :param numpy.array conf_scores: bounding box confidence scores. Shape = [num_boxes].
    :param numpy.array box_class: bounding box class scores.
    :param float nms_thres: Non-Max Suppression (NMS) IoU threshold. Lower confidence boxes that exceed this 
                            IoU threshold with the higher confidence boxes are suppressed.
    
    :return numpy.array nms_boxes: remaining boxes coordinates [ymin, xmin, ymax, xmax]. Shape = [num_boxes, 4].
    :return numpy.array nms_scores: remaining boxes confidence scores. Shape = [num_boxes].
    :return numpy.array nms_class: remaining boxes class scores.
    """
    
    # perform non-max suppression on the boxes
    nms_indices = tf.image.non_max_suppression(boxes = conf_boxes,
                                               scores = conf_scores,
                                               iou_threshold = nms_thres,
                                               max_output_size = 100)

    # extract the boxes that remain after nms
    nms_boxes = tf.gather(conf_boxes, nms_indices).numpy()
    nms_conf_scores = tf.gather(conf_scores, nms_indices).numpy()
    nms_box_cls = tf.gather(box_class, nms_indices).numpy()
    
    return nms_boxes, nms_conf_scores, nms_box_cls

def activate_and_scale_up_model_output(lbl, params):
    """
    Function to activates model output (after reshaping) and scales up model ouput to grid-relative.

    :param numpy.array lbl: Encoded predicted labels. Shape = (BATCH, GRID_H, GRID_W, N_ANCHORS, N_CLASSES + 5).
    :param dict params: Dictionary with hyperparameters.
    
    :param numpy.array lbl: activated and scaled labels. Shape = (BATCH, GRID_H, GRID_W, N_ANCHORS, N_CLASSES + 5).
    """
    
    # extract number of class labels & anchors
    N_CLASSES = tf.size(params['CLASSES'])
    N_ANCHORS = tf.shape(params['ANCHORS'])[0]

    # grid for top-left grid x coordinates
    cell_w_i = tf.repeat([tf.range(0, params['GRID_W'], 1)], params['GRID_H'], axis = 0) # [0 to (GRID_W-1)] as a row
    cell_w_i = tf.cast(cell_w_i, dtype = tf.float32) # cast to float data type
    cell_w_i = tf.repeat(cell_w_i, N_ANCHORS) # repeated for the number of anchors
    cell_w_i = tf.reshape(cell_w_i, (params['GRID_H'], params['GRID_W'], N_ANCHORS)) # reshaped 
    cell_w_i = tf.expand_dims(tf.expand_dims(cell_w_i, axis = 0), axis = -1) # [1, GRID_H, GRID_W, N_ANCHORS, 1]

    # grid for top-left grid y coordinates
    cell_h_i = tf.reshape(tf.range(0, params['GRID_H'], 1), [params['GRID_H'], 1]) # [0 to (GRID_H-1)] --> reshaped to a column
    cell_h_i = tf.cast(tf.repeat(cell_h_i, params['GRID_W'], axis = 1), dtype = tf.float32) # repeated for the number of horizontal grids 
    cell_h_i = tf.repeat(cell_h_i, N_ANCHORS) # repeated for the number of anchors
    cell_h_i = tf.reshape(cell_h_i, (params['GRID_H'], params['GRID_W'], N_ANCHORS)) # reshaped 
    cell_h_i = tf.expand_dims(tf.expand_dims(cell_h_i, axis = 0), axis = -1) # [1, GRID_H, GRID_W, N_ANCHORS, 1]

    # reshape grid-relative anchors for vectorized operations
    anchor_heights = tf.cast(tf.reshape(params['ANCHORS'][:,0], (1,1,1,N_ANCHORS,1)), tf.float32) # [1, 1, 1, ANCHORS, 1]
    anchor_widths  = tf.cast(tf.reshape(params['ANCHORS'][:,1], (1,1,1,N_ANCHORS,1)), tf.float32) # [1, 1, 1, ANCHORS, 1]

    # extract class probabilities
    class_prob = lbl[...,:N_CLASSES] # [BATCH, GRID_H, GRID_W, N_ANCHORS, N_CLASS]

    # extract and activate confidence score (sigmoid)
    conf_prob = tf.expand_dims(tf.sigmoid(lbl[...,N_CLASSES]), axis = -1) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 1]

    # extract and activate coordinates
    x_coord = tf.expand_dims(tf.sigmoid(lbl[...,tf.add(N_CLASSES, 1)]), axis = -1) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 1]
    y_coord = tf.expand_dims(tf.sigmoid(lbl[...,tf.add(N_CLASSES, 2)]), axis = -1) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 1]
    w_coord = tf.expand_dims(tf.exp(lbl[...,tf.add(N_CLASSES, 3)]), axis = -1) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 1]
    h_coord = tf.expand_dims(tf.exp(lbl[...,tf.add(N_CLASSES, 4)]), axis = -1) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 1]

    # scale up to grid relative predictions
    x_coord = tf.add(x_coord, cell_w_i) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 1]
    y_coord = tf.add(y_coord, cell_h_i) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 1]
    w_coord = tf.multiply(anchor_widths, w_coord) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 1]
    h_coord = tf.multiply(anchor_heights, h_coord) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 1]

    return tf.concat([class_prob, conf_prob, x_coord, y_coord, w_coord, h_coord], axis = -1) # [BATCH, GRID_H, GRID_W, N_ANCHORS, N_CLASS+5]
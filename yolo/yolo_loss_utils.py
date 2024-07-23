import tensorflow as tf

def minmax_convert(x, y, w, h):
    """
    Function to convert midpoint coordinates and width & height dimensions to
    bounding box min and max coordinates.
    
    :param tf.Tensor x: Midpoint x-cooordinates. 
    :param tf.Tensor y: Midpoint y-cooordinates.
    :param tf.Tensor w: Widths of bounding boxes.
    :param tf.Tensor h: Heights of bounding boxes.
    
    :return tf.Tensor x_min: Upperleft bounding box corner x-coordinates.
    :return tf.Tensor y_min: Upperleft bounding box corner y-coordinates.
    :return tf.Tensor x_max: Bottomright bounding box corner x-coordinates.
    :return tf.Tensor y_max: Bottomright bounding box corner y-coordinates.
    """
    x_min = tf.subtract(x, tf.divide(w, 2)) # [None, GRID_H, GRID_W, ANCHORS]
    y_min = tf.subtract(y, tf.divide(h, 2)) # [None, GRID_H, GRID_W, ANCHORS]
    x_max = tf.add(x, tf.divide(w, 2)) # [None, GRID_H, GRID_W, ANCHORS]
    y_max = tf.add(y, tf.divide(h, 2)) # [None, GRID_H, GRID_W, ANCHORS]
    
    return x_min, y_min, x_max, y_max 

def IoU(box1_xywh, box2_xywh):
    """
    Function to compute element-wise Intersection over Union (IoU).
    
    :param tf.Tensor box1_xywh: box1's x-mid, y-mid, width, height values. Shape = [BATCH, GRID_H, GRID_W, ANCHORS, 4].
    :param tf.Tensor box2_xywh: box2's x-mid, y-mid, width, height values. Shape = [BATCH, GRID_H, GRID_W, ANCHORS, 4].
    
    :return tf.Tensor ious: Intersection over Unions (IoUs). Shape = [BATCH, GRID_H, GRID_W, ANCHORS]
    """
    
    # convert box 1 xywh to xyminmax
    # Shape = [BATCH, GRID_H, GRID_W, ANCHORS]
    box1_x_min, box1_y_min, box1_x_max, box1_y_max = minmax_convert(x = box1_xywh[...,0], 
                                                                    y = box1_xywh[...,1],
                                                                    w = box1_xywh[...,2], 
                                                                    h = box1_xywh[...,3])
    
    # convert box 2 xywh to xyminmax
    # Shape = [BATCH, GRID_H, GRID_W, ANCHORS]
    box2_x_min, box2_y_min, box2_x_max, box2_y_max = minmax_convert(x = box2_xywh[...,0], 
                                                                    y = box2_xywh[...,1],
                                                                    w = box2_xywh[...,2], 
                                                                    h = box2_xywh[...,3])
    
    intersect_x_min = tf.maximum(box1_x_min, box2_x_min) # [BATCH, GRID_H, GRID_W, ANCHORS]
    intersect_y_min = tf.maximum(box1_y_min, box2_y_min) # [BATCH, GRID_H, GRID_W, ANCHORS]
    intersect_x_max = tf.minimum(box1_x_max, box2_x_max) # [BATCH, GRID_H, GRID_W, ANCHORS]
    intersect_y_max = tf.minimum(box1_y_max, box2_y_max) # [BATCH, GRID_H, GRID_W, ANCHORS]
    
    intersect_width  = tf.maximum(tf.subtract(intersect_x_max, intersect_x_min), 0.) # [BATCH, GRID_H, GRID_W, ANCHORS]
    intersect_height = tf.maximum(tf.subtract(intersect_y_max, intersect_y_min), 0.) # [BATCH, GRID_H, GRID_W, ANCHORS]
    
    intersect = tf.multiply(intersect_width, intersect_height) # [BATCH, GRID_H, GRID_W, ANCHORS]
    
    box1_w = box1_xywh[...,2] # [BATCH, GRID_H, GRID_W, ANCHORS]
    box1_h = box1_xywh[...,3] # [BATCH, GRID_H, GRID_W, ANCHORS]
    box2_w = box2_xywh[...,2] # [BATCH, GRID_H, GRID_W, ANCHORS]
    box2_h = box2_xywh[...,3] # [BATCH, GRID_H, GRID_W, ANCHORS]
    
    box_1_surface = tf.multiply(box1_w, box1_h) # [BATCH, GRID_H, GRID_W, ANCHORS]
    box_2_surface = tf.multiply(box2_w, box2_h) # [BATCH, GRID_H, GRID_W, ANCHORS]
    
    union = tf.subtract(tf.add(box_1_surface, box_2_surface), intersect) # [BATCH, GRID_H, GRID_W, ANCHORS]
    union = tf.maximum(union, 1e-5) # add a small number to union for numerical stability
    
    ious = tf.truediv(intersect, union) # [BATCH, GRID_H, GRID_W, ANCHORS]
    
    return ious

def IoU_matrix(box1_xywh, box2_xywh):
    """
    Function to compute Intersection over Union (IoU) matrix between two sets of (anchor) boxes.
    
    :param tf.Tensor box1_xywh: box1's x-mid, y-mid, width, height values. Shape = [BATCH, GRID_H, GRID_W, BOX1_ANCHORS, 4].
    :param tf.Tensor box2_xywh: box2's x-mid, y-mid, width, height values. Shape = [BATCH, GRID_H, GRID_W, BOX2_ANCHORS, 4].
    
    :return tf.Tensor iou: Intersection over Union (IoU) matrix. Shape = [BATCH, GRID_H, GRID_W, BOX1_ANCHORS, BOX2_ANCHORS]
    """
    
    # convert box 1 xywh to xyminmax
    # Shape = [BATCH, GRID_H, GRID_W, BOX1_ANCHORS]
    box1_x_min, box1_y_min, box1_x_max, box1_y_max = minmax_convert(x = box1_xywh[...,0], 
                                                                    y = box1_xywh[...,1],
                                                                    w = box1_xywh[...,2], 
                                                                    h = box1_xywh[...,3])
    
    # convert box 2 xywh to xyminmax
    # Shape = [BATCH, GRID_H, GRID_W, BOX2_ANCHORS]
    box2_x_min, box2_y_min, box2_x_max, box2_y_max = minmax_convert(x = box2_xywh[...,0], 
                                                                    y = box2_xywh[...,1],
                                                                    w = box2_xywh[...,2], 
                                                                    h = box2_xywh[...,3])
    
    box1_x_min = tf.expand_dims(box1_x_min, axis = -1) # [BATCH, GRID_H, GRID_W, BOX1_ANCHORS, 1]
    box1_y_min = tf.expand_dims(box1_y_min, axis = -1) # [BATCH, GRID_H, GRID_W, BOX1_ANCHORS, 1]
    box1_x_max = tf.expand_dims(box1_x_max, axis = -1) # [BATCH, GRID_H, GRID_W, BOX1_ANCHORS, 1]
    box1_y_max = tf.expand_dims(box1_y_max, axis = -1) # [BATCH, GRID_H, GRID_W, BOX1_ANCHORS, 1]
    
    box2_x_min = tf.expand_dims(box2_x_min, axis = -2) # [BATCH, GRID_H, GRID_W, 1, BOX2_ANCHORS]
    box2_y_min = tf.expand_dims(box2_y_min, axis = -2) # [BATCH, GRID_H, GRID_W, 1, BOX2_ANCHORS]
    box2_x_max = tf.expand_dims(box2_x_max, axis = -2) # [BATCH, GRID_H, GRID_W, 1, BOX2_ANCHORS]
    box2_y_max = tf.expand_dims(box2_y_max, axis = -2) # [BATCH, GRID_H, GRID_W, 1, BOX2_ANCHORS]
    
    intersect_x_min = tf.maximum(box1_x_min, box2_x_min) # [BATCH, GRID_H, GRID_W, BOX1_ANCHORS, BOX2_ANCHORS]
    intersect_y_min = tf.maximum(box1_y_min, box2_y_min) # [BATCH, GRID_H, GRID_W, BOX1_ANCHORS, BOX2_ANCHORS]
    intersect_x_max = tf.minimum(box1_x_max, box2_x_max) # [BATCH, GRID_H, GRID_W, BOX1_ANCHORS, BOX2_ANCHORS]
    intersect_y_max = tf.minimum(box1_y_max, box2_y_max) # [BATCH, GRID_H, GRID_W, BOX1_ANCHORS, BOX2_ANCHORS]
    
    intersect_width  = tf.maximum(tf.subtract(intersect_x_max, intersect_x_min), 0.) # [BATCH, GRID_H, GRID_W, BOX1_ANCHORS, BOX2_ANCHORS]
    intersect_height = tf.maximum(tf.subtract(intersect_y_max, intersect_y_min), 0.) # [BATCH, GRID_H, GRID_W, BOX1_ANCHORS, BOX2_ANCHORS]
        
    intersect = tf.multiply(intersect_width, intersect_height) # [BATCH, GRID_H, GRID_W, BOX1_ANCHORS, BOX2_ANCHORS]
    
    box1_w = tf.expand_dims(box1_xywh[...,2], axis = -1) # [BATCH, GRID_H, GRID_W, BOX1_ANCHORS, 1]
    box1_h = tf.expand_dims(box1_xywh[...,3], axis = -1) # [BATCH, GRID_H, GRID_W, BOX1_ANCHORS, 1]
    
    box2_w = tf.expand_dims(box2_xywh[...,2], axis = -2) # [BATCH, GRID_H, GRID_W, 1, BOX2_ANCHORS]
    box2_h = tf.expand_dims(box2_xywh[...,3], axis = -2) # [BATCH, GRID_H, GRID_W, 1, BOX2_ANCHORS]
    
    box_1_surface = tf.multiply(box1_w, box1_h) # [BATCH, GRID_H, GRID_W, BOX1_ANCHORS, 1]
    box_2_surface = tf.multiply(box2_w, box2_h) # [BATCH, GRID_H, GRID_W, 1, BOX2_ANCHORS]
    
    union = tf.subtract(tf.add(box_1_surface, box_2_surface), intersect) # [BATCH, GRID_H, GRID_W, BOX1_ANCHORS, BOX2_ANCHORS]
    union = tf.maximum(union, 1e-5) # add a small number to union for numerical stability
    
    ious = tf.truediv(intersect, union) # [BATCH, GRID_H, GRID_W, ANCHORS, ANCHORS]
    
    return ious

def get_true_boxes(lbl, params):
    """
    Function to extract all the ground truth bounding boxes within a batch and reshape for vectorized operations.
    
    :param tf.Tensor lbl: batch of y_true target labels. Shape = [BATCH, GRID_H, GRID_W, ANCHORS, (N_CLASSES+5)].
    :param dict params: hyperparameters.
    
    :return tf.Tensor true_boxes: true boxes. Shape = [BATCH, GRID_H, GRID_W, BUFFER, 4].
    """
    
    # [n_gt_obj_in_batch, 4] --> 4 elements for each batch are [batch_i, grid_h_i, grid_w_i, anchor_i]
    true_box_locs = tf.where(lbl[...,tf.size(params['CLASSES'])] == 1.)
    true_box_locs = true_box_locs[:params['BUFFER'],...]
    
    # true box coordinates
    true_boxes = tf.gather_nd(lbl, true_box_locs) # [n_gt_obj_in_batch, (n_classes + 5)]
    # isolate the box midpoints and dimensions
    true_boxes = true_boxes[...,-4:] # [n_gt_obj_in_batch, (n_classes + 5)]
    
    # extract batch indices
    batch_locs = true_box_locs[:,0]
    batch_indices = tf.expand_dims(tf.repeat(batch_locs, tf.shape(true_boxes)[1]), axis = -1)

    # column indices
    col_indices = tf.cast(tf.expand_dims(tf.reshape(tf.repeat(tf.expand_dims(tf.range(4), axis = 0), tf.shape(true_boxes)[0], axis = 0), [-1]), axis = -1), tf.int64)

    # row indices
    row_indices = tf.cast(tf.expand_dims(tf.repeat(tf.range(params['BUFFER']), 4)[:tf.cast(tf.multiply(tf.shape(true_boxes)[0], 4), tf.int32)], axis = -1), tf.int64)
   
    # create sparse tensor
    s_t = tf.sparse.SparseTensor(indices = tf.concat([batch_indices, 
                                                      row_indices, 
                                                      col_indices], axis = -1), # pass in concatenated indices
                                 values = tf.reshape(true_boxes, [-1]), # flatten the xywh coordinates
                                 dense_shape = [tf.cast(tf.shape(lbl)[0], tf.int64), # BATCH dim is dynamic (cannot be params['BATCH_SIZE']
                                                tf.cast(params['BUFFER'], tf.int64), 
                                                4]) # output shape = [BATCH, BUFFER, 4]
    
    # reorder indices
    t = tf.sparse.reorder(s_t)

    # sparse to dense tensor
    true_boxes = tf.sparse.to_dense(t) # output shape = [BATCH, BUFFER, 4]
    true_boxes = tf.repeat(tf.expand_dims(true_boxes, axis = 1), 
                           repeats = params['GRID_W'], axis = 1) # [BATCH, GRID_W, BUFFER, 4]
    true_boxes = tf.repeat(tf.expand_dims(true_boxes, axis = 1), 
                           repeats = params['GRID_H'], axis = 1) # [BATCH, GRID_H, GRID_W, BUFFER, 4]

    return true_boxes
	
def IoU_from_wh(box1_wh, box2_wh):
    """
    Function to compute Intersection over Union (IoU) between two boxes.
    
    :param tf.Tensor box1_hw: anchor boxes heights & widths [1, 1, 1, ANCHORS, 2].
    :param tf.Tensor box2_hw:    g_t boxes heights & widths [BATCH, GRID_H, GRID_W, ANCHORS, 2].
    
    :return float iou: Intersection over Union (IoU)
    """
    
    box1_wh = tf.expand_dims(box1_wh, axis = -2) # [1, 1, 1, ANCHORS, 1, 2]
    box2_wh = tf.expand_dims(box2_wh, axis = -3) # [BATCH, GRID_H, GRID_W, 1, ANCHORS, 2]
    
    intersect_h = tf.maximum(tf.minimum(box1_wh[...,0], box2_wh[...,0]), 0.) # [BATCH, GRID_H, GRID_W, ANCHORS, ANCHORS]
    intersect_w = tf.maximum(tf.minimum(box1_wh[...,1], box2_wh[...,1]), 0.) # [BATCH, GRID_H, GRID_W, ANCHORS, ANCHORS]
    intersect = tf.multiply(intersect_h, intersect_w)
    
    box1_area = tf.multiply(box1_wh[...,0], box1_wh[...,1]) # [BATCH, GRID_H, GRID_W, ANCHORS, ANCHORS]
    box2_area = tf.multiply(box2_wh[...,0], box2_wh[...,1]) # [BATCH, GRID_H, GRID_W, ANCHORS, ANCHORS]
    
    union = tf.subtract(tf.add(box1_area, box2_area), intersect) # [BATCH, GRID_H, GRID_W, ANCHORS, ANCHORS]
    union = tf.maximum(union, 1e-5)
    
    # ious of anchors (rows) with ground truth boxes (columns)
    iou = tf.truediv(intersect, union) # [BATCH, GRID_H, GRID_W, ANCHORS, ANCHORS]
    
    return iou
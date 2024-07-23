import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error, binary_crossentropy
from tensorflow.nn import softmax_cross_entropy_with_logits

from .yolo_loss_utils import IoU, IoU_matrix, get_true_boxes

def yolo_loss(params):
    
    def loss_function(y_true, y_pred):
        """
        Function to compute YOLO loss between predicted and ground truth labels.

        :param tf.Tensor y_true: ground truth label. Shape = [None, GRID_H, GRID_W, N_ANCHORS, N_CLASS+5].
        :param tf.Tensor y_pred: predicted label.    Shape = [None, GRID_H, GRID_W, N_ANCHORS, N_CLASS+5]

        :return tf.Tensor loss: loss value
        """
        
        # extract number of class labels & anchors
        N_CLASSES = tf.size(params['CLASSES'])
        N_ANCHORS = tf.shape(params['ANCHORS'])[0]
        
        ######### CONSTANTS #########
        lambda_coord = 1.0
        lambda_obj   = 1.0
        lambda_noobj = 1.0
        lambda_class = 1.0
        
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

        ######### PREDICTIONS #########
        
        # extract class probabilities
        # NOTE that we do not yet perform activation (softmax) here as the built in tensorflow cross entropy function
        # relies on unscaled logits as predictions for efficiency
        pred_class = y_pred[...,:N_CLASSES] # [BATCH, GRID_H, GRID_W, N_ANCHORS, N_CLASS]

        # extract and activate confidence score (sigmoid)
        pred_conf = tf.expand_dims(tf.sigmoid(y_pred[...,N_CLASSES]), axis = -1) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 1]

        # extract and activate coordinates
        x_coord = tf.expand_dims(tf.sigmoid(y_pred[...,tf.add(N_CLASSES, 1)]), axis = -1) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 1]
        y_coord = tf.expand_dims(tf.sigmoid(y_pred[...,tf.add(N_CLASSES, 2)]), axis = -1) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 1]
        # NOTE that we have to cap the input for tf.exp to avoid exceeding tf.float32 limits (returning inf/nan errors)
        w_coord = tf.expand_dims(tf.exp(tf.minimum(y_pred[...,tf.add(N_CLASSES, 3)], 80.)), axis = -1) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 1]
        h_coord = tf.expand_dims(tf.exp(tf.minimum(y_pred[...,tf.add(N_CLASSES, 4)], 80.)), axis = -1) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 1]

        # scale up to grid-relative [0 ~ GRID_H/W] predictions
        x_coord = tf.add(x_coord, cell_w_i) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 1]
        y_coord = tf.add(y_coord, cell_h_i) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 1]
        w_coord = tf.multiply(anchor_widths, w_coord) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 1]
        h_coord = tf.multiply(anchor_heights, h_coord) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 1]

        # concatenate mid-points & dimensions
        pred_xy = tf.concat([x_coord, y_coord], axis = -1) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 2]
        pred_wh = tf.concat([w_coord, h_coord], axis = -1) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 2]
        pred_xywh = tf.concat([pred_xy, pred_wh], axis = -1) # [BATCH, GRID_H, GRID_W, N_ANCHORS, 4]
        
        ######### GROUND TRUTH LABELS #########
        
        # extract ground truth (grid-relative [0 ~ GRID_H/W]) 
        true_class = y_true[...,:N_CLASSES] # [BATCH, GRID_H, GRID_W, ANCHORS, CLASS]
        true_conf = tf.expand_dims(y_true[...,N_CLASSES], axis = -1) # [BATCH, GRID_H, GRID_W, ANCHORS, 1]
        true_xywh = y_true[...,tf.add(N_CLASSES, 1):tf.add(N_CLASSES, 5)] # [BATCH, GRID_H, GRID_W, ANCHORS, 4]
        true_xy = true_xywh[...,0:2] # [BATCH, GRID_H, GRID_W, ANCHORS, 2]
        true_wh = true_xywh[...,2:]  # [BATCH, GRID_H, GRID_W, ANCHORS, 2]

        ######### MASKS & SCALING FACTORS #########
        
        # extract all ground truth boxes (objects) per image & reshape them for vectorized operations
        true_boxes = get_true_boxes(lbl = y_true, params = params) # [BATCH, GRID_H, GRID_W, BUFFERED_GT_BOXES, 4]
         
        # calculate IoU(predicted box, all ground truth boxes in the image) 
        # regardless of where the ground truth boxes appear in the image (not limited to the same grid as predicted box)
        ious = IoU_matrix(box1_xywh = pred_xywh, box2_xywh = true_boxes) # [BATCH, GRID_H, GRID_W, ANCHORS, BUFFERED_GT_BOXES]
        
        # find the best IoU for each predicted box with any ground truth box in the image
        best_ious = tf.reduce_max(ious, axis = -1) # [BATCH, GRID_H, GRID_W, ANCHORS]
        
        # object & (high and low IoU) no-object masks
        L_obj = true_conf[...,0] # [BATCH, GRID_H, GRID_W, ANCHORS]
        L_noobj_high_iou = tf.multiply(tf.cast(tf.greater_equal(best_ious, 0.6), tf.float32), tf.subtract(1.0, L_obj)) # [BATCH, GRID_H, GRID_W, ANCHORS]
        L_noobj_low_iou = tf.multiply(tf.subtract(1.0, L_noobj_high_iou), tf.subtract(1.0, L_obj)) # [BATCH, GRID_H, GRID_W, ANCHORS]
        
        # scaling factors
        # NOTE that we need at least 1 obj per image in the batch!
        N_obj = tf.maximum(tf.reduce_sum(L_obj, axis = [1,2,3]), 1.0) # [BATCH,]
        N_noobj_high_iou = tf.maximum(tf.reduce_sum(L_noobj_high_iou, axis = [1,2,3]), 1.0) # [BATCH,]
        N_noobj_low_iou = tf.maximum(tf.reduce_sum(L_noobj_low_iou, axis = [1,2,3]), 1.0) # [BATCH,]
        
        ######### LOCALIZATION LOSS #########
        
        xy_loss = tf.multiply(L_obj, mean_squared_error(y_true = true_xy, 
                                                        y_pred = pred_xy)) # [BATCH, GRID_H, GRID_W, ANCHORS]
        xy_loss = tf.reduce_sum(xy_loss, axis = [1,2,3]) # [BATCH,]
        wh_loss = tf.multiply(L_obj, mean_squared_error(y_true = tf.sqrt(true_wh), 
                                                        y_pred = tf.sqrt(pred_wh))) # [BATCH, GRID_H, GRID_W, ANCHORS]
        wh_loss = tf.reduce_sum(wh_loss, axis = [1,2,3]) # [BATCH,]
        
        local_loss = tf.multiply(tf.divide(lambda_coord, N_obj), tf.add(xy_loss, wh_loss)) # [BATCH,]
        
        ######### CLASS LOSS #########
        
        class_loss = softmax_cross_entropy_with_logits(labels = true_class, logits = pred_class) # [BATCH, GRID_H, GRID_W, ANCHORS]
        class_loss = tf.reduce_sum(tf.multiply(L_obj, class_loss), axis = [1,2,3]) # [BATCH,]
        class_loss = tf.multiply(tf.divide(lambda_class, N_obj), class_loss) # [BATCH,]
        
        ######### CONFIDENCE LOSS #########
        
        # object confidence loss
        conf_obj_loss = tf.multiply(L_obj, mean_squared_error(y_true = 1.0, y_pred = pred_conf)) # [BATCH, GRID_H, GRID_W, ANCHORS]
        conf_obj_loss = tf.reduce_sum(conf_obj_loss, axis = [1,2,3]) # [BATCH,]
        conf_obj_loss = tf.multiply(tf.divide(lambda_obj, N_obj), conf_obj_loss) # [BATCH,]
        
        # no-object high IoU confidence loss
        conf_noobj_high_iou_loss = tf.multiply(L_noobj_high_iou, mean_squared_error(y_true = tf.expand_dims(best_ious, axis = -1), 
                                                                                    y_pred = pred_conf)) # [BATCH, GRID_H, GRID_W, ANCHORS]
        conf_noobj_high_iou_loss = tf.reduce_sum(conf_noobj_high_iou_loss, axis = [1,2,3]) # [BATCH,]
        conf_noobj_high_iou_loss = tf.multiply(tf.divide(lambda_obj, N_noobj_high_iou), conf_noobj_high_iou_loss) # [BATCH,]
        
        # no-object low IoU confidence loss
        conf_noobj_low_iou_loss = tf.multiply(L_noobj_low_iou, mean_squared_error(y_true = 0.0, y_pred = pred_conf)) # [BATCH, GRID_H, GRID_W, ANCHORS]
        conf_noobj_low_iou_loss = tf.reduce_sum(conf_noobj_low_iou_loss, axis = [1,2,3]) # [BATCH,]
        conf_noobj_low_iou_loss = tf.multiply(tf.divide(lambda_noobj, N_noobj_low_iou), conf_noobj_low_iou_loss) # [BATCH,]
        
        # add the confidence loss terms
        conf_loss = tf.add_n([conf_obj_loss, conf_noobj_high_iou_loss, conf_noobj_low_iou_loss]) # [BATCH,]
        
        ######### TOTAL LOSS PER IMAGE #########

        loss = tf.add_n([local_loss, class_loss, conf_loss]) # [BATCH,]
        
        ######### AVERAGE LOSS PER BATCH #########
        
        loss = tf.reduce_mean(loss) # [1,]
            
        return loss
    
    return loss_function
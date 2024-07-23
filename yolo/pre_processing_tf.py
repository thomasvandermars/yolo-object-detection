import os
import cv2
import numpy as np
import tensorflow as tf

def check_for_duplicate_indices(t, obj_inds, anchor_meta_inds, anchor_inds):
    """
    Function that checks if we need to continue the while loop for removing duplicate index sets.
    
    :params tf.Tensor t: Loop tracker
    :params tf.Tensor obj_inds: Object index sets (GRID_H, GRID_W, ANCHOR_i). Shape = [n_bboxes, 3].
    :params tf.Tensor anchor_meta_inds: Locations of the anchor indices. Shape = [n_bboxes, 2].
    :params tf.Tensor anchor_inds: Anchor indices. Shape = [n_bboxes, n_anchors]. 
    
    :return tf.Tensor continue_loop: boolean for indicating whether we should continue to look to remove duplicates.
    """
    # group object index sets (GRID_H, GRID_W, ANCHOR_i) by converting to strings and concatenating each row
    inds_str = tf.strings.reduce_join(tf.strings.as_string(obj_inds), axis = -1, separator = "")
    
    # extract unique index sets
    inds_str_unique, unique_idx = tf.unique(inds_str)
    
    # duplicates exist when the number of unique sets is less than total number of index sets
    duplicates_exist = tf.less(tf.size(inds_str_unique), tf.size(inds_str))
    
    # check if the maximum number of loops haven't been reached (prevents infinite loops)
    max_n_loops_not_reached_yet = tf.less(t, 10)
    
    # we continue the loop when duplicates exist & we haven't reached the maximum number of loops
    continue_loop = tf.math.logical_and(duplicates_exist, max_n_loops_not_reached_yet)
    
    return continue_loop

def remove_duplicate_indices(t, obj_inds, anchor_meta_inds, anchor_inds):
    """
    Function that removes duplicate index sets.
    
    :params tf.Tensor t: Loop tracker
    :params tf.Tensor obj_inds: Object index sets (GRID_H, GRID_W, ANCHOR_i). Shape = [n_bboxes, 3].
    :params tf.Tensor anchor_meta_inds: Locations of the anchor indices. Shape = [n_bboxes, 2].
    :params tf.Tensor anchor_inds: Anchor indices. Shape = [n_bboxes, n_anchors]. 
    
    :return tf.Tensor t: Updated loop tracker
    :return tf.Tensor obj_inds: Updated object index sets (GRID_H, GRID_W, ANCHOR_i). Shape = [n_bboxes, 3].
    :return tf.Tensor anchor_meta_inds: Updated locations of the anchor indices. Shape = [n_bboxes, 2].
    :return tf.Tensor anchor_inds: Anchor indices. Shape = [n_bboxes, n_anchors].
    """
    
    # group object index sets (GRID_H, GRID_W, ANCHOR_i) by converting to strings and concatenating each row
    inds_str = tf.strings.reduce_join(tf.strings.as_string(obj_inds), axis = -1, separator = "")
    inds_str_unique, unique_idx = tf.unique(inds_str) # extract unique index sets

    # matrix: index sets (rows) vs. unique index sets (columns)
    matrix = tf.cast(tf.repeat(tf.expand_dims(inds_str, axis = -1), tf.size(inds_str_unique), -1) == inds_str_unique, tf.int32)
    # remove update range for the already unique index sets
    matrix = tf.multiply(matrix, tf.subtract(1, tf.cast(tf.math.cumsum(matrix, axis = 0) == 1, tf.int32)))
    
    # update meta indices to represent the new locations of the anchors selected 
    anchor_meta_inds = tf.stack([anchor_meta_inds[...,0], 
                                 tf.add(anchor_meta_inds[...,1], tf.reduce_sum(matrix, axis = -1))], axis = 1)
    
    # update objects indices
    obj_inds = tf.stack([obj_inds[...,0], # GRID_H indices 
                         obj_inds[...,1], # GRID_W indices 
                         tf.gather_nd(anchor_inds, anchor_meta_inds)], axis=1) # updated anchor indices (where anchor indices of duplcate sets are set to their next best option)

    # update loop tracker variable
    t = tf.add(t, 1)
    
    return t, obj_inds, anchor_meta_inds, anchor_inds

def random_contrast_brightness_hue(image, contrast = 0.5, brightness = 0.15, hue = 0.05):
    """
    Function that randomly adjusts contrast, brightness and hue for the inputted image.
    
    :params tf.Tensor image: input image
    :params tf.float32 contrast: factor for adjusting contast (1. = no adjustment in contrast).
    :params tf.float32 brightness: factor for adjusting brightness (0. = no adjustment in brightness). Delta should be in the range (-1,1).
    :params tf.float32 hue: factor for adjusting hue (0. = no adjustment in hue). Delta must be in the interval [-1, 1].
    
    :return tf.Tensor image: augmented image.
    """
    
    # randomly adjust contrast within specified range
    contrast_delta = tf.random.uniform(shape=[], minval = tf.subtract(1., contrast), maxval = tf.add(1., contrast), dtype=tf.float32)
    image = tf.image.adjust_contrast(image, contrast_delta)
    
    # randomly adjust brightness within specified range
    brightness_delta = tf.random.uniform(shape=[], minval = tf.multiply(-1., brightness), maxval = brightness, dtype=tf.float32)
    image = tf.image.adjust_brightness(image, brightness_delta)
    
    # randomly adjust hue within specified range
    hue_delta = tf.random.uniform(shape=[], minval = tf.multiply(-1., hue), maxval = hue, dtype=tf.float32)
    image = tf.image.adjust_hue(image, hue_delta)
    
    # as a result of the adjustments some values might fall outside the normalized range of [0,1]
    image = tf.clip_by_value(image, 0, 1.0)
    
    return image

def flip(image, boxes):
    """
    Function that flips the image horizontally and its corresponding bounding boxes.
    
    :params tf.Tensor img: image
    :params tf.Tensor boxes: bounding boxes [xmin, ymin, xmax, ymax]. Shape = [n_anchors, 4].
    
    :return tf.Tensor flipped_image: flipped image.
    :return tf.Tensor flipped_boxes: adjusted bounding box coordinates.
    """
    # we are flipping horizontally so the difference between the image's right border and the right edge of the bounding boxes
    # are the new xmin coordinates. ymin stays the same!
    xmins = tf.subtract(tf.subtract(tf.cast(tf.shape(image)[1], tf.float32), 1.), boxes[:,2])
    ymins = boxes[:,1]
    xmaxs = tf.add(xmins, tf.subtract(boxes[:,2], boxes[:,0]))
    ymaxs = tf.add(ymins, tf.subtract(boxes[:,3], boxes[:,1]))

    # concatenate the new coordinates
    flipped_boxes = tf.stack([xmins, ymins, xmaxs, ymaxs], axis = 1)

    # flip the image
    flipped_image = tf.image.flip_left_right(image)
    
    return flipped_image, flipped_boxes
    
def random_flip_left_right(image, boxes):
    """
    Function that randomly flips the image horizontally and its corresponding bounding boxes based on a randomly generated
    variable.
    
    :params tf.Tensor img: image
    :params tf.Tensor boxes: bounding boxes [xmin, ymin, xmax, ymax]. Shape = [n_anchors, 4].
    
    :return tf.Tensor img: returned image.
    :return tf.Tensor bxs: returned bounding box coordinates.
    """
    # if the random variable that is generated is between 0. and 0.5...
    img, bxs = tf.cond(tf.less(tf.random.uniform(shape = [1,], minval = 0., maxval = 1., dtype = tf.float32)[0], 0.5),
                       lambda: flip(image, boxes), # flip the image and the corresponding bounding boxes...
                       lambda: (image, boxes)) # otherwise, return the unflipped image and corresponding bounding boxes.
    
    return img, bxs

def random_translation(img, boxes, alpha):
    """
    Function randomly tranlates images and its corresponding bounding boxes.
    
    :params tf.Tensor img: image
    :params tf.Tensor boxes: bounding boxes [xmin, ymin, xmax, ymax]. Shape = [n_anchors, 4].
    :params tf.Tensor alpha: translate percentage [0.0, 0.5]
    
    :return tf.Tensor translated_img: adjusted image based on randomly generated translations.
    :return tf.Tensor translated_boxes: adjusted bounding box coordinates based on randomly generated translations.
    """
    
    # set boundaries on allowed translation
    alpha = tf.maximum(tf.minimum(alpha, 0.5), 0.0)
    
    # generate random offsets based on given translate percentage 
    x_offset = tf.random.uniform(shape = [1,], minval = tf.multiply(-1., alpha), maxval = alpha, dtype = tf.float32)[0]
    y_offset = tf.random.uniform(shape = [1,], minval = tf.multiply(-1., alpha), maxval = alpha, dtype = tf.float32)[0]
    
    # scale up to pixel level based on image dimensions
    img_h = tf.cast(tf.shape(img)[0], tf.float32)
    img_w = tf.cast(tf.shape(img)[1], tf.float32)
    x_offset = tf.round(tf.multiply(x_offset, img_w))
    y_offset = tf.round(tf.multiply(y_offset, img_h))
    
    # 1.) adjust bounding box coordinates based on offsets
    
    # if x offset is positive --> image is shifted to the right
    x_s = tf.cond(tf.greater(x_offset, 0.), 
              lambda: [tf.minimum(boxes[:,0], tf.subtract(tf.subtract(img_w, x_offset), 1.)),
                       tf.minimum(boxes[:,2], tf.subtract(tf.subtract(img_w, x_offset), 1.))], 
              lambda: [boxes[:,0], boxes[:,2]])

    # if y offset is positive --> image is shifted down
    y_s = tf.cond(tf.greater(y_offset, 0.), 
                  lambda: [tf.minimum(boxes[:,1], tf.subtract(tf.subtract(img_h, y_offset), 1.)),
                           tf.minimum(boxes[:,3], tf.subtract(tf.subtract(img_h, y_offset), 1.))], 
                  lambda: [boxes[:,1], boxes[:,3]])

    # if x offset is negative --> image is shifted to the left
    x_s = tf.cond(tf.less(x_offset, 0.), 
                  lambda: [tf.maximum(tf.add(x_s[0], x_offset), 0.),
                           tf.maximum(tf.add(x_s[1], x_offset), 0.)], 
                  lambda: x_s)

    # if y offset is negative --> image is shifted upwards
    y_s = tf.cond(tf.less(y_offset, 0.), 
                  lambda: [tf.maximum(tf.add(y_s[0], y_offset), 0.),
                           tf.maximum(tf.add(y_s[1], y_offset), 0.)], 
                  lambda: y_s)

    # concatenate new bounding box coordinates
    translated_boxes = tf.stack([x_s[0], y_s[0], x_s[1], y_s[1]], axis = 1)
    
    # 2.) adjust image based on offsets
    img = tf.cond(tf.greater(x_offset, 0.), lambda: img[:,:tf.cast(tf.subtract(img_w, x_offset), tf.int32),:], lambda: img)
    img = tf.cond(tf.greater(y_offset, 0.), lambda: img[:tf.cast(tf.subtract(img_h, y_offset), tf.int32),...], lambda: img)
    img = tf.cond(tf.less(x_offset, 0.), lambda: img[:,tf.cast(tf.multiply(x_offset, -1.), tf.int32):,:], lambda: img)
    img = tf.cond(tf.less(y_offset, 0.), lambda: img[tf.cast(tf.multiply(y_offset, -1.), tf.int32):,...], lambda: img)
    translated_img = img
    
    return translated_img, translated_boxes

def ranking_anchors_by_IoU(anchor_dims, bbox_dims):
    """
    Function that returns matrix with anchor indices sorted best to worst based on IoU with given bounding boxes.
    
    :params tf.Tensor anchor_dims: anchor dimensions [heights, widths]. Shape = [num_of_anchors, 2]
    :params tf.Tensor bbox_dims: bounding box dimensions [heights, widths] of the bounding boxes for which 
                                 the best anchors are sorted by IoU. Shape = [num_of_bboxes, 2]
    
    :return ordered_anchor_indices: anchor indices sorted best to worst IoU with a given bounding boxes. 
                                    Shape = [num_of_anchors, num_of_bboxes].
    """
    
    anchor_dims = tf.expand_dims(anchor_dims, axis = 1) # shape = [n_anchors, 1, 2]
    bbox_dims = tf.expand_dims(bbox_dims, axis = 0) # shape = [1, n_bboxes, 2]
    
    # find the minimum widths and heights
    intersect_height = tf.minimum(anchor_dims[...,0], bbox_dims[...,0]) # shape = [n_anchors, n_boxes]
    intersect_width  = tf.minimum(anchor_dims[...,1], bbox_dims[...,1]) # shape = [n_anchors, n_boxes]
    
    # calculate areas
    intersection = tf.multiply(intersect_width, intersect_height) # shape = [n_anchors, n_boxes]
    box_area = tf.expand_dims(tf.multiply(bbox_dims[0,:,0], bbox_dims[0,:,1]), axis = 0) # shape = [1, n_boxes]
    anchor_area = tf.expand_dims(tf.multiply(anchor_dims[:,0,0], anchor_dims[:,0,1]), axis = -1) # shape = [n_anchors, 1]
    
    # calculate IoU
    ious = tf.divide(intersection, tf.subtract(tf.add(box_area, anchor_area), intersection)) # shape = [n_anchors, n_boxes]
    
    # order the anchor indices from best to worst
    ordered_anchor_indices = tf.cast(tf.argsort(ious, direction = 'DESCENDING', axis = 0), tf.int32)
    
    return ordered_anchor_indices

def preprocess_tf(filename, tar_h, tar_w, grid_h, grid_w, channels, anchors, classes):
    """
    Function that preprocesses image-label pairings.
    
    :params tf.Tensor filname: path to image filename
    :params tf.Tensor tar_h: target height for image
    :params tf.Tensor tar_w: target width for image
    :params tf.Tensor grid_h: number of vertical grids that objects can be assigned to
    :params tf.Tensor grid_w: number of horizontal grids that objects can be assigned to
    :params tf.Tensor channels: number of color channels
    :params tf.Tensor anchors: anchor boxes (heights, widths). Shape = [n_anchors, 2]
    :params tf.Tensor classes: class labels
    
    :return img: preprocessed image. Shape = [IMG_H, IMG_W, 3].
    :return lbl: YOLO encoded label. Shape = [GRID_H, GRID_W, N_ANCHORS, N_CLASS+5].
    """
    
    # reading in image
    image = tf.io.read_file(filename)
    image = tf.io.decode_image(image, expand_animations = False, channels = channels) # NOTE that decoded image is normalized to [0,1] range
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # take image path and construct corresponding .txt annotation file path
    path = tf.strings.split(filename, '/')
    annotation_paths = tf.strings.join([tf.strings.reduce_join(path[:-2], separator = '/'), 'annotations (txt)', tf.strings.regex_replace(path[-1], '.png|.jpg|.jpeg', '.txt')], separator = '/')
    
    # read text annotation file
    label = tf.io.read_file(annotation_paths)
    label = tf.reshape(tf.strings.split(label), [-1, 5])
    
    # extract image dimensions, number of class labels and number of bounding boxes
    img_h = tf.cast(tf.shape(image)[0], tf.float32)
    img_w = tf.cast(tf.shape(image)[1], tf.float32)
    n_class = tf.size(classes)
    n_bboxes = tf.cast(tf.shape(label)[0], tf.int32)
    
    # extract bounding box coordinates
    coor = tf.strings.to_number(label[:,1:], tf.float32) # NOTE that we drop the first column which are the class labels
    
    # convert from absolute [xmin, ymin, xmax, ymax] to grid relative [xmid, ymid, width, height]
    w = tf.divide(tf.subtract(coor[:,2], coor[:,0]), img_w) # image relative bounding box widths
    h = tf.divide(tf.subtract(coor[:,3], coor[:,1]), img_h) # image relative bounding box heights
    x = tf.divide(tf.divide(tf.add(coor[:,2], coor[:,0]), 2.), img_w) # image relative bounding box x-midpoint
    y = tf.divide(tf.divide(tf.add(coor[:,3], coor[:,1]), 2.), img_h) # image relative bounding box y-midpoint

    x_box = tf.multiply(x, grid_w) # grid relative bounding box x-midpoint
    y_box = tf.multiply(y, grid_h) # grid relative bounding box y-midpoint
    w_box = tf.multiply(w, grid_w) # grid relative bounding box widths
    h_box = tf.multiply(h, grid_h) # grid relative bounding box widths

    grid_x = tf.cast(x_box, tf.int32) # x grid
    grid_y = tf.cast(y_box, tf.int32) # y grid
    
    # grid relative bounding box dimensions. Shape = [n_bboxes, 2]
    bbox_dims = tf.stack([h_box, w_box], axis = 1)
    
    # rank anchors by IoU with each bounding box output. Shape = [n_anchors, n_bboxes]
    ordered_anchor_indices = ranking_anchors_by_IoU(anchor_dims = tf.cast(anchors, tf.float32), # Shape = [n_anchors, 2]
                                                    bbox_dims = bbox_dims) # Shape = [n_bboxes, 2]
    ordered_anchor_indices = tf.transpose(ordered_anchor_indices) # [n_bboxes, n_anchors]
    
    # construct grid coordinate indices & the "best" anchor box assignment (GRID_H, GRID_W, ANCHOR_i). Shape = [n_bboxes, 3]
    indices = tf.stack([grid_y, grid_x, tf.cast(ordered_anchor_indices[...,0], tf.int32)], axis = 1)
     
    # meta indices for extracting the best anchors assigned to each bounding box
    idx = tf.stack([tf.range(0, tf.shape(ordered_anchor_indices)[0], 1), 
                    tf.cast(tf.zeros(tf.shape(ordered_anchor_indices)[0]), tf.int32)], axis = -1)
    
    # we need to eliminate any overlapping sets of grid-anchor indices
    tracker = tf.constant(0)
    tracker, indices, idx, ordered_anchor_indices = tf.while_loop(check_for_duplicate_indices, 
                                                                  remove_duplicate_indices, 
                                                                  loop_vars = [tracker, indices, idx, ordered_anchor_indices])
    
    # repeat the grid and anchor index sets 6 times (one for each bbox characteristic). Shape = [n_bboxes * 6, 3]
    indices = tf.repeat(indices, repeats = 6, axis = 0)
    
    # location indices for the objects
    cls_locs = tf.cast(tf.argmax(tf.equal(tf.expand_dims(label[:,0], axis = -1), tf.expand_dims(classes, axis = 0)), axis = 1), tf.int32)
    obj_locs = tf.cast(tf.repeat(n_class, n_bboxes), tf.int32)
    x_locs = tf.repeat(tf.cast(tf.add(n_class, 1), tf.int32), n_bboxes)
    y_locs = tf.repeat(tf.cast(tf.add(n_class, 2), tf.int32), n_bboxes)
    w_locs = tf.repeat(tf.cast(tf.add(n_class, 3), tf.int32), n_bboxes)
    h_locs = tf.repeat(tf.cast(tf.add(n_class, 4), tf.int32), n_bboxes)
    
    locs = tf.reshape(tf.stack([cls_locs, obj_locs, x_locs, y_locs, w_locs, h_locs], axis = 1), [-1])
    
    # add final column with indices
    indices = tf.concat([indices, tf.expand_dims(locs, axis = -1)], axis = -1)
    
    # object values to be stored in encoded label
    values = tf.reshape(tf.stack([tf.ones(n_bboxes), tf.ones(n_bboxes), x_box, y_box, w_box, h_box], axis = 1), [-1])
    
    # place the values within a sparse tensor
    s_t = tf.sparse.SparseTensor(indices = tf.cast(indices, tf.int64),
                                 values = values, 
                                 dense_shape = [grid_h, grid_w, tf.shape(anchors)[0], tf.add(tf.shape(classes)[0], 5)])

    # reorder indices
    t = tf.sparse.reorder(s_t)
    
    # sparse to dense tensor
    lbl = tf.sparse.to_dense(t)
    
    # finally we can resize the image
    img = tf.image.resize(image, [tar_h, tar_w])
    
    return img, lbl

def preprocess_with_augmentation_tf(filename, tar_h, tar_w, grid_h, grid_w, channels, anchors, classes, 
                                    translation = 0.2, contrast = 0.5, brightness = 0.15, hue = 0.05):
    """
    Function that preprocesses and augments image-label pairings.
    
    :params tf.Tensor filname: path to image filename
    :params tf.Tensor tar_h: target height for image
    :params tf.Tensor tar_w: target width for image
    :params tf.Tensor grid_h: number of vertical grids that objects can be assigned to
    :params tf.Tensor grid_w: number of horizontal grids that objects can be assigned to
    :params tf.Tensor channels: number of color channels
    :params tf.Tensor anchors: anchor boxes (heights, widths). Shape = [n_anchors, 2]
    :params tf.Tensor classes: class labels
    :params tf.Tensor translation: delta for randomly generating adjustments to image translations
    :params tf.Tensor contrast: delta for randomly generating adjustments to image contrast
    :params tf.Tensor brightness: delta for randomly generating adjustments to image brightness
    :params tf.Tensor hue: delta for randomly generating adjustments to image hue colors
    
    :return img: preprocessed image. Shape = [IMG_H, IMG_W, 3].
    :return lbl: YOLO encoded label. Shape = [GRID_H, GRID_W, N_ANCHORS, N_CLASS+5].
    """
    
    # reading in image
    image = tf.io.read_file(filename)
    image = tf.io.decode_image(image, expand_animations = False, channels = channels) # NOTE that decoded image is normalized to [0,1] range
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # take image path and construct corresponding .txt annotation file path
    path = tf.strings.split(filename, '/')
    annotation_paths = tf.strings.join([tf.strings.reduce_join(path[:-2], separator = '/'), 'annotations (txt)', tf.strings.regex_replace(path[-1], '.png|.jpg|.jpeg', '.txt')], separator = '/')
    
    # read text annotation file
    label = tf.io.read_file(annotation_paths)
    label = tf.reshape(tf.strings.split(label), [-1, 5])
    
    # extract number of class labels and number of bounding boxes
    n_class = tf.size(classes)
    n_bboxes = tf.cast(tf.shape(label)[0], tf.int32)
    
    # extract bounding box coordinates
    coor = tf.strings.to_number(label[:,1:], tf.float32) # NOTE that we drop the first column which are the class labels
    
    ###### AUGMENTATIONS #########
    
    # 1.) random translations
    image, coor = random_translation(img = image, boxes = coor, alpha = translation)
    
    # 2.) randomly flip horizontally (left to right)
    image, coor = random_flip_left_right(image = image, boxes = coor)
    
    # 3.) randomly adjust contrast, brightness and hue
    image = random_contrast_brightness_hue(image = image, contrast = contrast, brightness = brightness, hue = hue)
    
    ##############################
    
    # extract image shape (after augmentations were done)
    img_h = tf.cast(tf.shape(image)[0], tf.float32)
    img_w = tf.cast(tf.shape(image)[1], tf.float32)
    
    # convert from absolute [xmin, ymin, xmax, ymax] to grid relative [xmid, ymid, width, height]
    w = tf.divide(tf.subtract(coor[:,2], coor[:,0]), img_w) # image relative bounding box widths
    h = tf.divide(tf.subtract(coor[:,3], coor[:,1]), img_h) # image relative bounding box heights
    x = tf.divide(tf.divide(tf.add(coor[:,2], coor[:,0]), 2.), img_w) # image relative bounding box x-midpoint
    y = tf.divide(tf.divide(tf.add(coor[:,3], coor[:,1]), 2.), img_h) # image relative bounding box y-midpoint

    x_box = tf.multiply(x, grid_w) # grid relative bounding box x-midpoint
    y_box = tf.multiply(y, grid_h) # grid relative bounding box y-midpoint
    w_box = tf.multiply(w, grid_w) # grid relative bounding box widths
    h_box = tf.multiply(h, grid_h) # grid relative bounding box widths

    grid_x = tf.cast(x_box, tf.int32) # x grid
    grid_y = tf.cast(y_box, tf.int32) # y grid
    
    # grid relative bounding box dimensions. Shape = [n_bboxes, 2]
    bbox_dims = tf.stack([h_box, w_box], axis = 1)
    
    # rank anchors by IoU with each bounding box output. Shape = [n_anchors, n_bboxes]
    ordered_anchor_indices = ranking_anchors_by_IoU(anchor_dims = tf.cast(anchors, tf.float32), # Shape = [n_anchors, 2]
                                                    bbox_dims = bbox_dims) # Shape = [n_bboxes, 2]
    ordered_anchor_indices = tf.transpose(ordered_anchor_indices) # [n_bboxes, n_anchors]
    
    # construct grid coordinate indices & the "best" anchor box assignment (GRID_H, GRID_W, ANCHOR_i). Shape = [n_bboxes, 3]
    indices = tf.stack([grid_y, grid_x, tf.cast(ordered_anchor_indices[...,0], tf.int32)], axis = 1)
     
    # meta indices for extracting the best anchors assigned to each bounding box
    idx = tf.stack([tf.range(0, tf.shape(ordered_anchor_indices)[0], 1), 
                    tf.cast(tf.zeros(tf.shape(ordered_anchor_indices)[0]), tf.int32)], axis = -1)
    
    # we need to eliminate any overlapping sets of grid-anchor indices
    tracker = tf.constant(0)
    tracker, indices, idx, ordered_anchor_indices = tf.while_loop(check_for_duplicate_indices, 
                                                                  remove_duplicate_indices, 
                                                                  loop_vars = [tracker, indices, idx, ordered_anchor_indices])
    
    # repeat the grid and anchor index sets 6 times (one for each bbox characteristic). Shape = [n_bboxes * 6, 3]
    indices = tf.repeat(indices, repeats = 6, axis = 0)
    
    # location indices for the objects
    cls_locs = tf.cast(tf.argmax(tf.equal(tf.expand_dims(label[:,0], axis = -1), tf.expand_dims(classes, axis = 0)), axis = 1), tf.int32)
    obj_locs = tf.cast(tf.repeat(n_class, n_bboxes), tf.int32)
    x_locs = tf.repeat(tf.cast(tf.add(n_class, 1), tf.int32), n_bboxes)
    y_locs = tf.repeat(tf.cast(tf.add(n_class, 2), tf.int32), n_bboxes)
    w_locs = tf.repeat(tf.cast(tf.add(n_class, 3), tf.int32), n_bboxes)
    h_locs = tf.repeat(tf.cast(tf.add(n_class, 4), tf.int32), n_bboxes)
    
    locs = tf.reshape(tf.stack([cls_locs, obj_locs, x_locs, y_locs, w_locs, h_locs], axis = 1), [-1])
    
    # add final column with indices
    indices = tf.concat([indices, tf.expand_dims(locs, axis = -1)], axis = -1)
    
    # object values to be stored in encoded label
    values = tf.reshape(tf.stack([tf.ones(n_bboxes), tf.ones(n_bboxes), x_box, y_box, w_box, h_box], axis = 1), [-1])
    
    # place the values within a sparse tensor
    s_t = tf.sparse.SparseTensor(indices = tf.cast(indices, tf.int64),
                                 values = values, 
                                 dense_shape = [grid_h, grid_w, tf.shape(anchors)[0], tf.add(tf.shape(classes)[0], 5)])

    # reorder indices
    t = tf.sparse.reorder(s_t)
    
    # sparse to dense tensor
    lbl = tf.sparse.to_dense(t)
    
    # finally we can resize the image
    img = tf.image.resize(image, [tar_h, tar_w])
    
    return img, lbl
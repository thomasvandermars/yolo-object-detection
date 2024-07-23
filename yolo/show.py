import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from .draw import draw_box, draw_box_label, draw_grid, draw_midpoint, draw_title
from .post_processing import decode, non_max_suppression, activate_and_scale_up_model_output

def show_sample(xml_filename, img_filename, formatting):
    """
    Function to show image and the corresponding bounding boxes described in the .xml annotation file.
    
    :param str xml_filename: xml filename of the annotation file to be shown.
    :param dict bbox_formatting: Dictionary with bounding box style options.
    
    :return: None
    """
    
    # make some assertions about the parameters
    assert(isinstance(formatting, dict))
    
    # extract XML file root
    tree = ET.parse(xml_filename)
    root = tree.getroot()

    # read in image
    img = cv2.imread(img_filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    
    # iterate through objects in annotation files
    for i in root.findall('object'):

        xmin = float(i.find('bndbox').find('xmin').text)
        ymin = float(i.find('bndbox').find('ymin').text)
        xmax = float(i.find('bndbox').find('xmax').text)
        ymax = float(i.find('bndbox').find('ymax').text)

        # draw bounding box
        draw_box(xmin = xmin, 
                 ymin = ymin, 
                 width = (xmax - xmin), 
                 height = (ymax - ymin), 
                 color = formatting['box_border_color'],
                 borderwidth = formatting['box_border_linewidth'],
                 borderstyle = formatting['box_border_linestyle'],
                 alpha = formatting['alpha'],
                 fill = formatting['box_filled_in'])
        
        # draw object label
        draw_box_label(xmin = xmin, 
                       ymin = ymin, 
                       label = i.find('name').text,
                       fontsize = formatting['label_fontsize'],
                       color = formatting['label_color'], 
                       backgroundcolor = formatting['label_background_color'],
                       bordercolor = formatting['label_border_color'],
                       borderwidth = formatting['label_border_width'],
                       borderstyle = formatting['label_border_style'],
                       padding = formatting['label_padding'])
                       
        pass
    pass
    
def show_preprocessed_sample(img, lbl, params, formatting):
    """
    Function to show preprocessed image-label pairing.
    
    :param numpy.array img: Image data. Shape = (batch, image_height, image_width, channels).
    :param numpy.array lbl: Encoded target labels. Shape = (batch, GRID_H, GRID_W, classes + 5)
    :param dict params: Dictionary with hyperparameters.
    :param dict bbox_formatting: Dictionary with bounding box style options.
    
    :return: None
    """

    # make some assertions about the parameters
    assert(isinstance(params, dict))
    assert(isinstance(formatting, dict))
    
    # repeat color channel for black and white images
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis = -1)
    
    # show image
    plt.cla()
    plt.imshow(img)
    
    # draw grid
    draw_grid(params = params, 
              color = formatting['grid_color'],
              linestyle = formatting['grid_linestyle'],
              alpha = formatting['grid_line_alpha'])
    
    title = [] # list to hold title elements
    
    # decode the YOLO-type label encoding
    # NOTE that we set confidence threshold to 1.0 because we are looking at target encodings (not at model predictions)
    boxes, confs, cls, inds = decode(x = lbl, params = params, conf_thres = 1.0)
    
    # iterate through object/bounding boxes
    for i in range(len(confs)):
        
        # draw bounding box
        draw_box(xmin = boxes[i,1], 
                 ymin = boxes[i,0], 
                 width = boxes[i,3] - boxes[i,1], 
                 height = boxes[i,2] - boxes[i,0],
                 color = formatting['box_border_color'],
                 borderwidth = formatting['box_border_linewidth'],
                 borderstyle = formatting['box_border_linestyle'],
                 alpha = formatting['alpha'],
                 fill = formatting['box_filled_in'])
        
        # draw midpoint coordinates
        draw_midpoint(xmid = boxes[i,1] + ((boxes[i,3] - boxes[i,1])/2), 
                      ymid = boxes[i,0] + ((boxes[i,2] - boxes[i,0])/2),  
                      color = formatting['box_border_color'])
        
        # draw object label
        draw_box_label(xmin = boxes[i,1], 
                       ymin = boxes[i,0],
                       label = params['CLASSES'][np.argmax(cls[i])],
                       fontsize = formatting['label_fontsize'],
                       color = formatting['label_color'], 
                       backgroundcolor = formatting['label_background_color'],
                       bordercolor = formatting['label_border_color'],
                       borderwidth = formatting['label_border_width'],
                       borderstyle = formatting['label_border_style'],
                       padding = formatting['label_padding'])
        
        # update title list
        title.append('Label: ' + params['CLASSES'][np.argmax(cls[i])] + ' - [Row: ' + str(inds[i,0] + 1) + ' / Col: ' + str(inds[i,1] + 1) + ']')
        pass
    
    # construct the title
    draw_title(title)
    pass

def show_prediction_vs_ground_truth(x, 
                                    y, 
                                    y_hat, 
                                    conf_thres, 
                                    nms_thres, 
                                    params, 
                                    formatting, 
                                    gt_bbox_formatting, 
                                    pred_bbox_formatting,
                                    axis_drawn = True):
    """
    Function to show predictive performance.
    
    :param numpy.array x: Image data. Shape = (image_height, image_width, channels).
    :param numpy.array y: Encoded target label. Shape = (GRID_H, GRID_W, classes + 5)
    :param numpy.array y_hat: Encoded predicted label. Shape = (GRID_H, GRID_W, classes + anchors * 5).
    :param float conf_thres: Object Confidence threshold. Bounding boxes with object confidence lower than 
                             this threshold are filtered out.
    :param float nms_thres: Non-Max Suppression (NMS) IoU threshold. Lower confidence boxes that exceed this 
                            IoU threshold with the higher confidence boxes are suppressed.
    :param dict params: Dictionary with hyperparameters.
    :param dict gt_bbox_formatting: Dictionary with ground truth bounding box style options.
    :param dict pred_bbox_formatting: Dictionary with predicted bounding box style options.
    
    :return: None
    """

    # make some assertions about the parameters
    assert(conf_thres >= 0.)
    assert(conf_thres <= 1.)
    assert(nms_thres >= 0.)
    assert(nms_thres <= 1.)
    assert(isinstance(params, dict))
    assert(isinstance(gt_bbox_formatting, dict))
    assert(isinstance(pred_bbox_formatting, dict))
    
    # show image
    plt.cla()
    plt.imshow(x, cmap='gray')
    
    # list to hold title elements
    title = []
    
    ####### plot grid ########
    
    draw_grid(params = params, 
              color = formatting['grid_color'],
              linestyle = formatting['grid_linestyle'],
              alpha = formatting['grid_line_alpha'])
    
    ####### plot ground truth boxes ########
    
    # decode the YOLO-type label encoding for ground truth label (confidence threshold = 1.0)
    boxes, confs, cls, inds = decode(x = y, params = params, conf_thres = 1.0)
    
    # iterate through object/bounding boxes
    for i in range(len(confs)):
        
        # draw bounding box
        draw_box(xmin = boxes[i,1], 
                 ymin = boxes[i,0], 
                 width = boxes[i,3] - boxes[i,1], 
                 height = boxes[i,2] - boxes[i,0],
                 color = gt_bbox_formatting['box_border_color'],
                 borderwidth = gt_bbox_formatting['box_border_linewidth'],
                 borderstyle = gt_bbox_formatting['box_border_linestyle'],
                 alpha = gt_bbox_formatting['alpha'],
                 fill = gt_bbox_formatting['box_filled_in'])
        
        # draw midpoint coordinates
        draw_midpoint(xmid = boxes[i,1] + ((boxes[i,3] - boxes[i,1])/2), 
                      ymid = boxes[i,0] + ((boxes[i,2] - boxes[i,0])/2),  
                      color = gt_bbox_formatting['box_border_color'])
        
        # draw object label
        if gt_bbox_formatting['label_fontsize'] > 0.0:
            draw_box_label(xmin = boxes[i,1], 
                           ymin = boxes[i,0],
                           label = params['CLASSES'][np.argmax(cls[i])],
                           fontsize = gt_bbox_formatting['label_fontsize'],
                           color = gt_bbox_formatting['label_color'], 
                           backgroundcolor = gt_bbox_formatting['label_background_color'],
                           bordercolor = gt_bbox_formatting['label_border_color'],
                           borderwidth = gt_bbox_formatting['label_border_width'],
                           borderstyle = gt_bbox_formatting['label_border_style'],
                           padding = gt_bbox_formatting['label_padding'])
        
        # update title list
        title.append('Label: ' + params['CLASSES'][np.argmax(cls[i])] + ' - [Row: ' + str(inds[i,0] + 1) + ' / Col: ' + str(inds[i,1] + 1) + ']')
        pass
    
    ####### plot predicted boxes ########
    
    # filter out predicted boxes that do not exceed confidence threshold
    boxes, confs, cls, inds = decode(x = y_hat, params = params, conf_thres = conf_thres)
    
    # apply non-max suppression to the confidence boxes
    nms_boxes, nms_conf_scores, nms_box_cls = non_max_suppression(boxes, confs, cls, nms_thres)
    
    # iterate through nms boxes
    for i in range(nms_boxes.shape[0]):
        
        # draw bounding box
        draw_box(xmin = nms_boxes[i,1], 
                 ymin = nms_boxes[i,0], 
                 width = nms_boxes[i,3]-nms_boxes[i,1], 
                 height = nms_boxes[i,2]-nms_boxes[i,0], 
                 color = pred_bbox_formatting['box_border_color'],
                 borderwidth = pred_bbox_formatting['box_border_linewidth'],
                 borderstyle = pred_bbox_formatting['box_border_linestyle'],
                 alpha = pred_bbox_formatting['alpha'],
                 fill = pred_bbox_formatting['box_filled_in'])
        
        # draw midpoint coordinates
        draw_midpoint(xmid = (nms_boxes[i,3]-nms_boxes[i,1])/2.0 + nms_boxes[i,1], 
                      ymid = (nms_boxes[i,2]-nms_boxes[i,0])/2.0 + nms_boxes[i,0],  
                      color = pred_bbox_formatting['box_border_color'])
        
        # draw object label
        if pred_bbox_formatting['label_fontsize'] > 0.0:
            draw_box_label(xmin = nms_boxes[i,1], 
                           ymin = nms_boxes[i,0],
                           label = params['CLASSES'][np.argmax(nms_box_cls[i])],
                           fontsize = pred_bbox_formatting['label_fontsize'],
                           color = pred_bbox_formatting['label_color'], 
                           backgroundcolor = pred_bbox_formatting['label_background_color'],
                           bordercolor = pred_bbox_formatting['label_border_color'],
                           borderwidth = pred_bbox_formatting['label_border_width'],
                           borderstyle = pred_bbox_formatting['label_border_style'],
                           padding = pred_bbox_formatting['label_padding'])
        
        # update title list
        title.append(('Y_hat (conf ' + str(np.round(nms_conf_scores[i] * 100.0, 2)) + '%): ' + params['CLASSES'][np.argmax(nms_box_cls[i])] + 
                      ' (' + str(np.round(np.max(nms_box_cls[i]) * 100.0, 2)) + '%)'))
        
        pass
    
    # construct the title
    draw_title(title)
    
    # remove axis values if specified
    if axis_drawn != True:
        plt.axis('off')
        pass
    
    return boxes, nms_boxes
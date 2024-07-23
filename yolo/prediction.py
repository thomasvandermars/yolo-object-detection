import cv2
import numpy as np
import os
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from .draw import draw_box, draw_box_label, draw_grid, draw_midpoint, draw_title
from .post_processing import decode, non_max_suppression, activate_and_scale_up_model_output

def predict_image(filename, 
                  model, 
                  conf_thres, 
                  nms_thres, 
                  params, 
                  formatting = {'box_border_color': 'fuchsia',
                                'box_border_linewidth': 2,
                                'box_border_linestyle': '-',
                                'box_filled_in': False,
                                'alpha': None,
                                'label_fontsize': 10.0,
                                'label_color': 'white',
                                'label_background_color': 'fuchsia',
                                'label_border_color': 'fuchsia',
                                'label_border_width': 2,
                                'label_border_style': '-',
                                'label_padding': None,
                                'grid_color': 'gray',
                                'grid_linestyle': '--',
                                'grid_line_alpha': 0.5,
                                'label_include_conf_score': False}, 
                  return_original_dimensions = True,
                  axis_drawn = True):
    """
    Function to predict and show the prediction on new "unseen" image data.
    
    :param str filename: image filename in test/image/ folder.
    :param tensorflow.keras.Model model: Trained model used to make the predictions.
    :param float conf_thres: Object Confidence threshold. Bounding boxes with object confidence lower than 
                             this threshold are filtered out.
    :param float nms_thres: Non-Max Suppression (NMS) IoU threshold. Lower confidence boxes that exceed this 
                            IoU threshold with the higher confidence boxes are suppressed.
    :param dict params: Dictionary with hyperparameters.
    :param dict formatting: Dictionary with formatting parameters.
    :param bool return_original_dimensions: Boolean variable indicating whether the image and the predicted 
                                            bounding boxes are scaled back to the image's original dimensions. 
                                            True means the image is scaled back to its original dimensions.
    :param bool axis_drawn: True if axis should be drawn around the image.
    
    :return numpy.array conf_boxes: bounding boxes that exceed the object confidence threshold [ymin, xmin, ymax, xmax]. Shape = (n, 4).
    :return numpy.array nms_boxes: non-max suppressed bounding boxes [ymin, xmin, ymax, xmax]. Shape = (n, 4).
    """

    # make some assertions about the parameters
    assert(conf_thres >= 0.)
    assert(conf_thres <= 1.)
    assert(nms_thres >= 0.)
    assert(nms_thres <= 1.)
    assert(isinstance(params, dict))
    assert(isinstance(formatting, dict))
    assert(isinstance(return_original_dimensions, bool))

    # read in image
    img_org = cv2.cvtColor(cv2.imread('test/image/' + filename), cv2.COLOR_BGR2RGB)
    img_org_h, img_org_w, _ = img_org.shape
    width_rescale_factor, height_rescale_factor = 1., 1.

    if return_original_dimensions == True:
        width_rescale_factor = img_org_w / params['IMG_W']
        height_rescale_factor = img_org_h / params['IMG_H']
        pass
    
    # resize image to require input size
    img_input = cv2.resize(img_org, (params['IMG_W'], params['IMG_H']))
    
    # use the provided model to make a prediction (NOTE that the input has to be standardize to [0,1] range)
    y_hat = model.predict(np.expand_dims(img_input / 255., axis = 0), verbose = 0)
    y_hat = activate_and_scale_up_model_output(lbl = y_hat, params = params)
    
    # show image
    target_h = None
    plt.cla()
    if return_original_dimensions:
        plt.imshow(img_org)
        target_h = img_org.shape[0]
    else:
        plt.imshow(img_input)
        target_h = img_input.shape[0]
        pass
    
    # draw grid
    #draw_grid()
    
    title = [] # list to hold title elements
    
    # filter out predicted boxes that do not exceed confidence threshold
    boxes, confs, cls, inds = decode(x = y_hat[0], params = params, conf_thres = conf_thres)
    
     # apply non-max suppression to the confidence boxes
    nms_boxes, nms_conf_scores, nms_box_cls = non_max_suppression(boxes, confs, cls, nms_thres)
    
    # iterate through nms boxes
    for i in range(nms_boxes.shape[0]):
        
        # draw bounding box
        draw_box(xmin = nms_boxes[i,1]*width_rescale_factor, 
                 ymin = nms_boxes[i,0]*height_rescale_factor, 
                 width = (nms_boxes[i,3]-nms_boxes[i,1])*width_rescale_factor, 
                 height = (nms_boxes[i,2]-nms_boxes[i,0])*height_rescale_factor,
                 color = formatting['box_border_color'],
                 borderwidth = formatting['box_border_linewidth'],
                 borderstyle = formatting['box_border_linestyle'],
                 alpha = formatting['alpha'],
                 fill = formatting['box_filled_in'])
        
        # draw midpoint coordinates
        draw_midpoint(xmid = ((nms_boxes[i,3]-nms_boxes[i,1])/2.0 + nms_boxes[i,1])*width_rescale_factor, 
                      ymid = ((nms_boxes[i,2]-nms_boxes[i,0])/2.0 + nms_boxes[i,0])*height_rescale_factor,  
                      color = formatting['box_border_color'])
        
        # construct bounding box label text
        txt = params['CLASSES'][np.argmax(nms_box_cls[i])] + ' (' + str(np.round(np.max(nms_box_cls[i]) * 100.0, 0)) + '%)'
        if formatting['label_include_conf_score']:
            txt = txt + ' - [cf: ' + str(np.round(nms_conf_scores[i] * 100.0, 0)) + '%]'
            pass
        
        # draw object label
        draw_box_label(xmin = (nms_boxes[i,1]*width_rescale_factor), 
                       ymin = (nms_boxes[i,0]*height_rescale_factor),
                       label = txt,
                       fontsize = formatting['label_fontsize'],
                       color = formatting['label_color'], 
                       backgroundcolor = formatting['label_background_color'],
                       bordercolor = formatting['label_border_color'],
                       borderwidth = formatting['label_border_width'],
                       borderstyle = formatting['label_border_style'],
                       padding = formatting['label_padding'])
        pass
       
    # construct the title
    draw_title(title)
    
    # remove axis values if specified
    if axis_drawn != True:
        plt.axis('off')
        pass
    
    plt.title(filename)
    
    return boxes, nms_boxes

def predict_video(filename, 
                  model, 
                  conf_thres, 
                  nms_thres, 
                  params, 
                  fps = 25,
                  folder = 'test/video/',
                  formatting = {'box_border_color': (255, 0, 0),
                                'box_border_thickness': 2,
                                'label_font': cv2.FONT_HERSHEY_SIMPLEX,
                                'label_font_size': 1,
                                'label_font_color': (255, 255, 255),
                                'label_font_thickness': 2,
                                'label_border_color': (255, 0, 0),
                                'label_border_thickness': 2,
                                'label_background_color':(255, 0, 0),
                                'label_include_conf_score': False}):
    """
    Function to use the provided model to do object detection on the frames of a video.
    And then return a reconstructed video with the bounding boxes.
    
    :param str filename: Image filename.
    :param tf.keras.model model: YOLOv2 model.
    :param float conf_thres: confidence threshold. Predictions that have an object confidence score lower than this threshold
                             are excluded.
    :param float nms_thres: non-max supression threshold. Lower confidence boxes than have IoU exceeding this threshold are "pruned" (excluded).
    :param dict params: Hyperparameters.
    :param int fps: frames per second. Defaults to 25.
    :param dict formatting: dictionary with formatting options. See default for formatting options.
    
    :return list frame_inference_times: List with the model inference times for each frame.
    """
    
    
    # make the directory for the results if it does not exist already
    if not os.path.exists(folder + 'results'):
        os.makedirs(folder + 'results')
        pass
    
    # open video that we want to do object detection on and get the number of frames
    cap = cv2.VideoCapture(folder + filename)
    amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    h = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff)
    
    # establish video dimensions
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    res, frame = cap.read()
    height, width, _ = frame.shape
    
    # open video file to write to
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for .mp4 format
    video = cv2.VideoWriter(folder + 'results/' + filename.split('.')[-2] + '_(YOLO).mp4', fourcc, fps, (width, height))

    # list to store frame prediction times
    frame_inference_times = []
    
    print('Creating video (.mp4)...')
    
    # iterate through frames
    for i in tqdm(range(int(amount_of_frames))):
        
        ############## read in frame and resize to input dimensions ###############
        
        # read in frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        res, frame = cap.read()
        
        # use object detector model to make prediction on frame
        start = time.time() # start time frame inference
        #------------
        
        # read in image
        img_org = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_org_h, img_org_w, _ = img_org.shape
        width_rescale_factor = img_org_w / params['IMG_W']
        height_rescale_factor = img_org_h / params['IMG_H']

        # resize image to require input size
        img_input = cv2.resize(img_org, (params['IMG_W'], params['IMG_H']))

        ############## predict frame using YOLO model ###############
        
        # use the provided model to make a prediction
        y_hat = model.predict(np.expand_dims(img_input / 255., axis = 0), verbose = 0)
        y_hat = activate_and_scale_up_model_output(lbl = y_hat, params = params)
        
        # filter out predicted boxes that do not exceed confidence threshold
        boxes, confs, cls, inds = decode(x = y_hat[0], params = params, conf_thres = conf_thres)
        
        # apply non-max suppression to the confidence boxes
        nms_boxes, nms_conf_scores, nms_box_cls = non_max_suppression(boxes, confs, cls, nms_thres)
        
        # convert a copy of the image back to bgr color channels
        img_org_bgr = cv2.cvtColor(img_org.copy(), cv2.COLOR_RGB2BGR)
        
        ############## Drawing Boxes ##############
        
        # iterate through nms boxes
        for i in range(nms_boxes.shape[0]):

            # extract and rescale bounding box coordinates
            xmin = int(nms_boxes[i,1] * width_rescale_factor)
            ymin = int(nms_boxes[i,0] * height_rescale_factor)
            xmax = int(nms_boxes[i,3] * width_rescale_factor)
            ymax = int(nms_boxes[i,2] * height_rescale_factor)
            
            # set text for label
            txt = str(params['CLASSES'][np.argmax(nms_box_cls[i])] + ' [' + str(np.round(np.max(nms_box_cls[i]) * 100.0, 2)) + '%]')
            if formatting['label_include_conf_score']:
                txt = txt + ' - [cf: ' + str(np.round(nms_conf_scores[i] * 100.0, 0)) + '%]'
                pass
            
            # retrieve text dimensions
            text_size, _ = cv2.getTextSize(text = txt, 
                                           fontFace = formatting['label_font'], 
                                           fontScale = formatting['label_font_size'], 
                                           thickness = formatting['label_font_thickness'])
            text_width, text_height = text_size
                
            # draw bounding box around object
            cv2.rectangle(img = img_org_bgr,
                          pt1 = (xmin, ymin),
                          pt2 = (xmax, ymax),
                          color = formatting['box_border_color'], 
                          thickness = formatting['box_border_thickness'])

            # draw box for background of class label text
            cv2.rectangle(img = img_org_bgr,
                          pt1 = (xmin, (ymin - text_height - 20)),
                          pt2 = ((xmin + text_width + 5), ymin),
                          color = formatting['label_background_color'], 
                          thickness = -1)
                
            # draw box border for class label 
            cv2.rectangle(img = img_org_bgr,
                          pt1 = (xmin, (ymin - text_height - 20)),
                          pt2 = ((xmin + text_width + 5), ymin),
                          color = formatting['label_border_color'], 
                          thickness = formatting['label_border_thickness'])
                
            # plot text for class label
            cv2.putText(img = img_org_bgr, 
                        text = txt, 
                        org = (xmin + 5, ymin - 10), 
                        fontFace = formatting['label_font'], 
                        fontScale = formatting['label_font_size'], 
                        color = formatting['label_font_color'],
                        thickness = formatting['label_font_thickness'])
            pass
        
        # convert back to BGR channels and write to video
        video.write(img_org_bgr)
        pass
        
        #------------
        end = time.time() # end time frame inference
        
        frame_inference_times.append(end - start) # store frame inference time
    
    # release the video objects
    cap.release()
    video.release()
    #cv2.destroyAllWindows()
    
    return frame_inference_times
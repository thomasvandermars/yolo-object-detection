import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from .anchor_box_clustering import ranking_anchors_by_IoU

def preprocess_np(image, label, params):
    """
    Function to load and resize images. Also, the corresponding annotation file is read in 
    and the bounding box coordinates encoded into YOLO (v2) target format.
    
    :param str image: Path to image file.
    :param str label: Path to annotation file.
    :param dict params: Dictionary with hyperparameters.
    
    :return numpy.array img_array: Numpy array with preprocess image.
    :return numpy.array lbl_array: Numpy array with encoded label.
    """
    # assertions
    assert(isinstance(params, dict))

    # assert valid parameter values
    assert(image.split('/')[-1][:-4] == label.split('/')[-1][:-4])
    
    # load in image and convert to image array
    img = cv2.imread(image) # read in image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB
    img = cv2.resize(img, (params['IMG_W'], params['IMG_H']))
    img_array = np.asarray(img) # convert to numpy array
    
    # infer class list length & number of anchors
    CLASS = len(params['CLASSES'])
    ANCHOR = params['ANCHORS'].shape[0]
    
    # initialize label zero array
    lbl_array = np.zeros([params['GRID_H'], params['GRID_W'], ANCHOR, CLASS + 5])
    
    # extract XML file root
    tree = ET.parse(label)
    root = tree.getroot()

    # extract original image dimensions
    img_h, img_w = int(root.find('size').find('height').text), int(root.find('size').find('width').text)
    
    # iterate through objects in annotation files
    for i in root.findall('object'):

        # if the detected object has a class label we are interested in
        if i.find('name').text in params['CLASSES']:
                
            # extract label & coordinates
            class_index = params['CLASSES'].index(i.find('name').text)
            xmin = float(i.find('bndbox').find('xmin').text)
            ymin = float(i.find('bndbox').find('ymin').text)
            xmax = float(i.find('bndbox').find('xmax').text)
            ymax = float(i.find('bndbox').find('ymax').text)

            # image relative width, height, and midpoint coordinates (x,y)
            w = (xmax - xmin) / img_w # width relative to picture width [0,1]
            h = (ymax - ymin) / img_h # height relative to picture height [0,1]
            x = (xmin + xmax) / 2 / img_w # mid-point x-coordinate relative to picture width [0,1]
            y = (ymin + ymax) / 2 / img_h # mid-point y-coordinate relative to picture height [0,1]

            # grid relative midpoint coordinates
            x_box = x * params['GRID_W'] # mid-point x-coordinate relative to picture grid width [0, GRID_W]
            y_box = y * params['GRID_H'] # mid-point y-coordinate relative to picture grid height [0, GRID_H]
            w_box = w * params['GRID_W'] # bbox width relative to picture grid width [0, GRID_W]
            h_box = h * params['GRID_H'] # bbox height relative to picture grid height [0, GRID_H]
            
            # extract grid offsets from upper left corner
            grid_x = int(x_box) 
            grid_y = int(y_box)

            # rank the anchors by IoU
            ordered_anchor_indices = ranking_anchors_by_IoU(anchor_dims = params['ANCHORS'], bbox_dims = (h_box, w_box))
            
            # iterate through the anchor IDs ranked by IoU
            for next_best_anchor_id in ordered_anchor_indices:
                
                # if we have not already assigned an object present to the grid, we can assign one...
                # else we look to assign the object to a next best fitting anchor 
                if lbl_array[grid_y, grid_x, next_best_anchor_id, CLASS] == 0:
                    lbl_array[grid_y, grid_x, next_best_anchor_id, class_index] = 1 # assign 1 to at location of class
                    lbl_array[grid_y, grid_x, next_best_anchor_id, CLASS] = 1 # assign 1 to indicate that object is present in this grid
                    lbl_array[grid_y, grid_x, next_best_anchor_id, (CLASS+1):(CLASS+5)] = [x_box, y_box, w_box, h_box]
                    break # once we assigned
                    pass
                pass
            pass
        pass
    
    return img_array/255., lbl_array

def preprocess_with_augmentation_np(image, label, params, augment_params):
    """
    Function to load and augment image (by adjusting color, exposure, saturation, etc.). The image is also randomly translated 
    (randomly cropped) so that the objects appear in different sections (grids) of the image. These augmentations will help 
    prevent overfitting on the training images. The annotation file is read in, the bounding box coordinates adjusted based
    on the translation and then encoded into YOLO (v2) target format.
    
    :param str image: Path to image file.
    :param str label: Path to annotation file.
    :param dict params: Dictionary with hyperparameters.
    :param dict augment_params: Dictionary with maximum degrees of translation, hue, contrast, 
                                brightness, and saturation being applied
    
    :return numpy.array img_array: Numpy array with pixel values of the augmented image.
    :return numpy.array lbl_array: Numpy array with encoded label (adjusted for augmentation).
    """
    
    # assert valid parameter values
    assert(image.split('/')[-1][:-4] == label.split('/')[-1][:-4])
    assert(isinstance(params, dict))
    assert(isinstance(augment_params, dict))
    
    # load in image and convert to image array
    img = cv2.imread(image) # read in image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB
    img_array = np.asarray(img) # convert to numpy array
    
    # infer class list length & number of anchors
    CLASS = len(params['CLASSES'])
    ANCHOR = params['ANCHORS'].shape[0]
    
    # initialize label zero array
    lbl_array = np.zeros([params['GRID_H'], params['GRID_W'], ANCHOR, CLASS + 5])
    
    # iterate through objects in annotation files and store them as bounding box objects
    bboxes = []
    tree = ET.parse(label)
    root = tree.getroot()
    
    for i in root.findall('object'):

        # if the detected object has a class label we are interested in
        if i.find('name').text in params['CLASSES']:
            
            bboxes.append(BoundingBox(x1 = float(i.find('bndbox').find('xmin').text), 
                                      y1 = float(i.find('bndbox').find('ymin').text), 
                                      x2 = float(i.find('bndbox').find('xmax').text), 
                                      y2 = float(i.find('bndbox').find('ymax').text),
                                      label = int(params['CLASSES'].index(i.find('name').text))))
            pass
        pass
    
    # rescale image and the corresponding bounding boxes
    bbs = BoundingBoxesOnImage(bboxes, shape=img_array.shape)
    image_rescaled = ia.imresize_single_image(img_array, (params['IMG_H'], params['IMG_W']))
    bbs_rescaled = bbs.on(image_rescaled)
    
    # create an augmentation pipeline
    seq = iaa.Sequential([iaa.Fliplr(0.5),
                          iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, augment_params['blur']))),        
                          iaa.LinearContrast((1.0-augment_params['contrast'], 1.0+augment_params['contrast'])),
                          iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, augment_params['noise'] * 255), per_channel=0.5),
                          iaa.Multiply((1.0-augment_params['brightness'], 1.0+augment_params['brightness']), per_channel=0.2),
                          iaa.Affine(scale={"x": (1.0-augment_params['zoom'], 1.0+augment_params['zoom']),
                                            "y": (1.0-augment_params['zoom'], 1.0+augment_params['zoom'])},
                                     translate_percent={"x": (-1*augment_params['translate'], augment_params['translate']),
                                                        "y": (-1*augment_params['translate'], augment_params['translate'])})
                          # rotate=(-25, 25),
                          # shear=(-8, 8))
                           ],random_order=True)

    # augment the resized image en the corresponding bounding boxes
    image_aug, bbs_aug = seq(image = image_rescaled, bounding_boxes = bbs_rescaled)
        
    # clip the bounding box values if they exceed the image boundaries
    bbs_aug = bbs_aug.clip_out_of_image()
    
    # iterate through the augmented bounding boxes and encode the YOLO label
    for i in range(len(bbs_aug.bounding_boxes)):
        
        # extract label & coordinates
        class_index = bbs_aug.bounding_boxes[i].label
        xmin = float(bbs_aug.bounding_boxes[i].x1)
        ymin = float(bbs_aug.bounding_boxes[i].y1)
        xmax = float(bbs_aug.bounding_boxes[i].x2)
        ymax = float(bbs_aug.bounding_boxes[i].y2)
        
        # image relative width, height, and midpoint coordinates (x,y)
        w = (xmax - xmin) / params['IMG_W'] # width relative to picture width [0,1]
        h = (ymax - ymin) / params['IMG_H'] # height relative to picture height [0,1]
        x = (xmin + xmax) / 2 / params['IMG_W'] # mid-point x-coordinate relative to picture width [0,1]
        y = (ymin + ymax) / 2 / params['IMG_H'] # mid-point y-coordinate relative to picture height [0,1]
        
        # grid relative midpoint coordinates
        x_box = x * params['GRID_W'] # mid-point x-coordinate relative to picture grid width [0, GRID_W]
        y_box = y * params['GRID_H'] # mid-point y-coordinate relative to picture grid height [0, GRID_H]
        w_box = w * params['GRID_W'] # bbox width relative to picture grid width [0, GRID_W]
        h_box = h * params['GRID_H'] # bbox height relative to picture grid height [0, GRID_H]
        
        # extract grid offsets from upper left corner
        grid_x = int(x_box) 
        grid_y = int(y_box)

        # rank the anchors by IoU
        ordered_anchor_indices = ranking_anchors_by_IoU(anchor_dims = params['ANCHORS'], bbox_dims = (h_box, w_box))
            
        # iterate through the anchor IDs ranked by IoU
        for next_best_anchor_id in ordered_anchor_indices:
            
            # if we have not already assigned an object present to the grid, we can assign one...
            # else we look to assign the object to a next best fitting anchor 
            if lbl_array[grid_y, grid_x, next_best_anchor_id, CLASS] == 0:
                lbl_array[grid_y, grid_x, next_best_anchor_id, class_index] = 1 # assign 1 to at location of class
                lbl_array[grid_y, grid_x, next_best_anchor_id, CLASS] = 1 # assign 1 to indicate that object is present in this grid
                lbl_array[grid_y, grid_x, next_best_anchor_id, (CLASS+1):(CLASS+5)] = [x_box, y_box, w_box, h_box]
                break # once we assigned
                pass
            pass
        pass
    
    return image_aug/255., lbl_array
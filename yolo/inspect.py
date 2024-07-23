import os
import xml.etree.ElementTree as ET
import random
from tqdm import tqdm

def sample_width_to_height_ratios(xml_filenames, sample_size):
    """
    Function to calculate the image width-to-height ratios of a sample from the dataset.
    
    :param list xml_filenames: list with all the xml annotation files in the dataset
    :param int sample_size: size of sample
    
    :return list width_to_height_ratios: list with the sampled width-to-height ratios
    """
    
    # assertions
    assert(isinstance(xml_filenames, list))
    assert(sample_size > 0)
    
    # cap the sample size at the size of the dataset
    if sample_size > len(xml_filenames):
        sample_size = len(xml_filenames)
        pass
    
    # list to store the extracted width to height ratios
    width_to_height_ratios = []
    
    # iterate through sampled annotation files
    for xml_file in tqdm(random.sample(xml_filenames, sample_size)):
        
        # extract XML file root
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # compute the width to height ratio
        width = float(root.find('size').find('width').text)
        height = float(root.find('size').find('height').text)
        width_to_height_ratios.append(width/height)
        pass
        
    return width_to_height_ratios
 
def sample_class_label_distribution(xml_filenames, sample_size, classes):
    """
    Function to sample the class label distribution of the dataset.
    
    :param list xml_filenames: list with all the xml annotation files in the dataset
    :param int sample_size: size of sample
    :param list classes: list of class labels to be included in the class label distribution 
    
    :return dict class_distribution: dictionary with the class label distribution
    """
    # assertions
    assert(isinstance(xml_filenames, list))
    assert(sample_size > 0)
    
    # cap the sample size at the size of the dataset
    if sample_size > len(xml_filenames):
        sample_size = len(xml_filenames)
        pass
    
    class_distribution = {}
    obj_count = 0
    
    # iterate through sampled annotation files
    for xml_file in tqdm(random.sample(xml_filenames, sample_size)):
        
        # extract XML file root
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # iterate through objects in annotation files
        for i in root.findall('object'):
            
            # if the detected object has a class label we are interested in
            if i.find('name').text in classes:
            
                if i.find('name').text in class_distribution.keys():
                    class_distribution[i.find('name').text] += 1
                else:
                    class_distribution[i.find('name').text] = 1
                    pass
                
                obj_count += 1 # increment the object count
                pass
            pass
    
    # make the values relative to sample size
    for cl in class_distribution.keys():
        class_distribution[cl] = class_distribution[cl]/obj_count
        pass
        
    return class_distribution

def get_class_labels(xml_filenames):
    """
    Function to get all unique object class labels in the dataset.
    
    :param list xml_filenames: list with all the xml annotation files in the dataset
    
    :return list classes: list of unique class labels for the objects in the dataset
    """
    
    # assertions
    assert(isinstance(xml_filenames, list))
    
    classes = [] # list to hold all the unique class labels
    
    # iterate through sampled annotation files
    for xml_file in tqdm(xml_filenames):
        
        # extract XML file root
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # iterate through objects in annotation files
        for i in root.findall('object'):
            
            if i.find('name').text not in classes:
                classes.append(i.find('name').text)
                pass
            pass
        pass
    
    # sort the class list for consistency
    classes.sort()
    
    return classes

def get_bounding_box_dimensions(xml_filenames, classes):
    """
    Function to get all bounding box dimensions (widths and heights) for the given class labels in the dataset.
    
    :param list xml_filenames: list with all the xml annotation files in the dataset.
    :param list classes: list of class labels to be included, other existing class labels are filtered out.
    
    :return list bbox_widths: list of bounding box widths.
    :return list bbox_heights: list of bounding box heights.
    """
    
    # initialize the objects to hold the information
    bbox_widths, bbox_heights = [], []
    
    # iterate through sampled annotation files
    for xml_file in tqdm(xml_filenames):

        # extract XML file root
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # compute the width to height ratio
        width = float(root.find('size').find('width').text)
        height = float(root.find('size').find('height').text)
        
        # iterate through objects in annotation files
        for i in root.findall('object'):
            
            if i.find('name').text in classes: 
                
                xmin = float(i.find('bndbox').find('xmin').text)
                ymin = float(i.find('bndbox').find('ymin').text)
                xmax = float(i.find('bndbox').find('xmax').text)
                ymax = float(i.find('bndbox').find('ymax').text)
                
                bbox_widths.append((xmax - xmin) / width)
                bbox_heights.append((ymax - ymin) / height)
                
                pass
            pass
        pass
    
    return bbox_widths, bbox_heights

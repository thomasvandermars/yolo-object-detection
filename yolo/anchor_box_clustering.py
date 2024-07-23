import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

def mean_IoU_by_number_of_anchors(n, bbox_widths, bbox_heights):
    """
    Function to evaluate mean IoU between the anchors and the bounding boxes for up to n anchors.
    
    :param int n: number of anchors up to which we want explore the mean IoU between the anchors and the bounding boxes.
    :params list bbox_widths: list of bounding box widths.
    :params list bbox_heights: list of bounding box heights.
    
    :return mean_ious: list of mean IoU's between the anchors and the bounding boxes for each subset of n anchors.
    """
    
    # assertions
    assert(n > 0)
    assert(isinstance(bbox_widths, list))
    assert(isinstance(bbox_heights, list))
    
    # create a zip list, tying the corresponding bbox widths and heights
    bboxes = list(zip(bbox_heights, bbox_widths))
    
    # list to store the mean Intersection over Unions (IoUs)
    mean_ious = []
    
    # iterate over the number of anchors we want to explore
    for c_n in tqdm(range(1, n+1)):

        # find the cluster centerpoints for widths and heights
        kmeans = KMeans(n_clusters = c_n, n_init = 10)
        kmeans.fit(bboxes)
        
        # cluster centerpoints (widths, heights)
        clusters = kmeans.cluster_centers_
        # cluster indices assigned to each bbox
        assigned_cluster_index = kmeans.labels_

        iou = [] # list to temporarily store bbox IoU with its assigned cluster center 

        # iterate through the bounding boxes
        for i in range(len(bboxes)):

            # find the minimum height between the bbox and the assigned cluster
            intersect_height = min(clusters[assigned_cluster_index[i],0], bboxes[i][0])
            # find the minimum width between the bbox and the assigned cluster
            intersect_width = min(clusters[assigned_cluster_index[i],1], bboxes[i][1])
            
            # intersection area
            intersection = intersect_width * intersect_height
            # bounding box area
            box_area = bboxes[i][0] * bboxes[i][1]
            # assigned cluster area
            cluster_area = clusters[assigned_cluster_index[i],0] * clusters[assigned_cluster_index[i],1]

            # add Intersection over Union (IoU)
            iou.append(intersection / (box_area + cluster_area - intersection))
            pass

        # calculate and store mean Intersection over Union (IoU) for the given number of clusters
        mean_ious.append(np.mean(np.array(iou)))
        pass
    
    return mean_ious

def find_anchors(n, bbox_heights, bbox_widths):
    """
    Run the kmeans algorithm on the bounding box dimensions and find the cluster center coordinates (dimensions)
    for each of the n clusters
    
    :param int n: number of dimension clusters.
    :params list bbox_widths: list of bounding box widths.
    :params list bbox_heights: list of bounding box heights.
    
    :return clusters: dimension clusters [height, width].
    """
    
    # assertions
    assert(n > 0)
    assert(isinstance(bbox_widths, list))
    assert(isinstance(bbox_heights, list))
    
    kmeans = KMeans(n_clusters = n, n_init = 10, random_state = 5)
    kmeans.fit(list(zip(bbox_heights, bbox_widths)))
    
    return kmeans.cluster_centers_, kmeans.labels_

def ranking_anchors_by_IoU(anchor_dims, bbox_dims):
    """
    Function that returns the indices of the anchors sorted best to worst based on IoU with a given bounding box.
    
    :params numpy.array anchor_dims: anchor dimensions [heights, widths]. Shape = [num_of_anchors, 2]
    :params tuple bbox_dims: bounding box dimensions (heights, widths) of the bounding box for which the best anchors are sorted by IoU.
    
    :return ordered_anchor_indices: anchor indices sorted best to worst IoU with a given bounding box.
    """
    
    # find the minimum widths and heights
    intersect_height = np.minimum(anchor_dims[:,0], bbox_dims[0])
    intersect_width = np.minimum(anchor_dims[:,1], bbox_dims[1])
    
    # calculate areas
    intersection = intersect_width * intersect_height
    box_area = bbox_dims[0] * bbox_dims[1]
    anchor_area = anchor_dims[:,0] * anchor_dims[:,1]
    
    # calculate IoU
    ious = intersection / (box_area + anchor_area - intersection)
    
    # order the anchor indices from best to worst
    ordered_anchor_indices = np.flip(ious.argsort())
    
    return ordered_anchor_indices
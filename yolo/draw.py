import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

def draw_box(xmin, 
             ymin, 
             width, 
             height, 
             color, 
             alpha, 
             borderwidth, 
             borderstyle = '-', 
             fill = False):
    """
    Function to draw a bounding box on top of an image.
    
    :param float/int xmin: Top left x coordinate of bounding box.
    :param float/int ymin: Top left y coordinate of bounding box.
    :param float/int width: Width of bounding box.
    :param float/int height: Height of bounding box.
    :param str color: Name of color to be used for the bounding box.
    :param str alpha: transparancy level of bounding box fill.
    :param str borderwidth: Width of the bounding box border.
    :param str borderstyle: Bounding box border style. Default is '-'.
    :param boolean fill: True is we want the bounding box filled in with color.
    
    :return: None
    """
	# assertions
    assert(color in list(mcolors.CSS4_COLORS.keys()))
	
    # depending on if we want the bounding box filled in...
    if fill:
        plt.gca().add_patch(Rectangle((xmin, ymin), 
                                       width, height, 
                                       linewidth = borderwidth,
                                       linestyle = borderstyle,
                                       color = color, 
                                       alpha = alpha))
    else:
        plt.gca().add_patch(Rectangle((xmin, ymin), 
                                       width, height, 
                                       linewidth = borderwidth,
                                       linestyle = borderstyle,
                                       facecolor = 'none',                                       
                                       edgecolor = color))
    pass
    
def draw_midpoint(xmid, ymid, color):
    """
    Function to draw a bounding box midpoint coordinate on top of an image.
    
    :param float/int xmid: Middle x coordinate of bounding box.
    :param float/int ymid: Middle y coordinate of bounding box.
    :param str color: Name of color to be used for the midpoint coordinate.
    
    :return: None
    """
	# assertions
    assert(color in list(mcolors.CSS4_COLORS.keys()))
	
    plt.scatter(xmid, ymid, c = color)
    pass

def draw_box_label(xmin, 
                   ymin, 
                   label, 
                   fontsize = 10.0, 
                   color = 'blue', 
                   backgroundcolor = 'none', 
                   bordercolor = 'none', 
                   borderwidth = 2, 
                   borderstyle = '-',
                   padding = 5.0):
    """
    Function to draw a bounding box class label on top of an image.
    
    :param float/int xmin: Top left x coordinate of bounding box.
    :param float/int ymin: Top left y coordinate of bounding box.
    :param str label: Name of class label.
    :param float fontsize: Fontsize label. Default is 10.0.
    :param str color: Name of color to be used for the class label. Default is 'blue'.
    :param str backgroundcolor: Name of color to be used for the background of the class label. Default is 'none'.
    :param str bordercolor: Name of color to be used for the border of the class label. Default is 'none'.
    :param int borderwidth: Borderwidth of the class label. Default is 2.
    :param str borderstyle: Bounding box border style. Default is '-'.
    :param float padding: Amount of padding between label string and border. Default is 5.0.
	
    :return: None
    """
	# assertions
    assert(color in list(mcolors.CSS4_COLORS.keys()))
    
    plt.text(x = xmin + 5, 
             y = ymin - 5, 
             s = label, 
             color = color, 
             size = fontsize,
             verticalalignment = 'bottom',
             bbox = dict(facecolor = backgroundcolor, 
                         edgecolor = bordercolor, 
                         pad = padding, 
                         linewidth = borderwidth, 
                         linestyle = borderstyle))   
    pass

def draw_grid(params, color, linestyle, alpha):
    """
    Function to draw grid lines that correspond to the grids used in the YOLO algorithm.
    
    :param dict params: hyperparameter dictionary.
    
    :return: None
    """
    # assertions
    assert(isinstance(params, dict))
    
    # establish grid dimensions
    GRID_HEIGHT = params['IMG_H'] / params['GRID_H']
    GRID_WIDTH = params['IMG_W'] / params['GRID_W']
    
    # vertical gridlines
    v_t = GRID_HEIGHT
    while v_t < params['IMG_H']-1:
        plt.axhline(y = v_t, ls = linestyle, color = color, alpha = alpha)
        v_t += GRID_HEIGHT
        pass
    
    # horizontal gridlines
    h_t = GRID_WIDTH
    while h_t < params['IMG_W']-1:
        plt.axvline(x = h_t, ls = linestyle, color = color, alpha = alpha)
        h_t += GRID_WIDTH
        pass
    pass

def draw_title(title):
    """
    Function to construct a multi-line title for an image.
    
    :param list title: List that holds the elements that make up the title.
    
    :return: None
    """
    
    # assertions
    assert(isinstance(title, list))
    
    t = '' # str variable that will become the multi-line title
    
    for i in range(len(title)): # iterate through title elements
        if i < len(title)-1: # for all but last element...
            t += title[i] + '\n' # we append the string element + next line character
        else: # for the last element..
            t += title[i] # we only append the string element
    
    # add the title to the image
    plt.title(t)
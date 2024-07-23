import os
import shutil
from tqdm import tqdm
import xml.etree.ElementTree as ET

def create_txt_annotations():
    """
    Function that creates corresponding text annotations for each Pascal VOC XML annotations.
    """
    # iterate over datasets....
    for dataset in os.listdir(os.getcwd() + '/datasets/'):

        # if the dataset's txt annotations do not exist...
        if not os.path.exists(os.getcwd() + '/datasets/' + dataset + '/annotations (txt)'):
            print('creating new [' + dataset + '] folder for text annotations...')
            # create a folder for them...
            os.makedirs(os.getcwd() + '/datasets/' + dataset + '/annotations (txt)')
            pass
        # if the folder does exist...
        else:
            try:
                # delete it...
                print('delete existing [' + dataset + '] folder with text annotations...')
                shutil.rmtree(os.getcwd() + '/datasets/' + dataset + '/annotations (txt)')
                # and create a new empty folder
                print('creating new [' + dataset + '] folder for text annotations...')
                os.makedirs(os.getcwd() + '/datasets/' + dataset + '/annotations (txt)')
            except:
                print('a problem occured when trying to delete existing text annotations...')
                break
        
        print('creating text annotations for the [' + dataset + '] folder...')
        
        # iterate through xml annotations...
        for xml_ann_file in os.listdir(os.getcwd() + '/datasets/' + dataset + '/annotations'):

            # extract XML file root
            tree = ET.parse(os.getcwd() + '/datasets/' + dataset + '/annotations/' + xml_ann_file)
            root = tree.getroot()

            obj = []

            # iterate through objects in annotation file
            for i in root.findall('object'):
                # create txt line representing a single object
                obj.append(i.find('name').text + ' ' + 
                           i.find('bndbox').find('xmin').text + ' ' +
                           i.find('bndbox').find('ymin').text + ' ' +
                           i.find('bndbox').find('xmax').text + ' ' +
                           i.find('bndbox').find('ymax').text + '\n')
                pass

            # open up txt file and write the object lines
            with open(os.getcwd() + '/datasets/' + dataset + '/annotations (txt)/' + xml_ann_file[:-4] + '.txt', "w") as file:
                for o in obj:
                    file.write(o)
                pass

            pass

def combine_datasets(datasets):
    """
    Function to combine the image filenames (.jpg & .png) and the annotation filenames (.xml) from different datasets within the data folder.
    The datasets folders all have an image subfolder holding the images and an annotation subfolder holding the .xml files.
    
    :param list datasets: List of dataset names within the data folder we would like to combine for model training & evaluation

    :return list X: List of pathnames to the image files.
    :return list Y: list of pathnames to the (.xml) annotation files. 
    """
    
    # combine the datasets while taking into account the different file formats (.jpg .png) of the images
    X, Y = [], []
    for i in range(len(datasets)):
        X += [os.getcwd() + '/datasets/' + datasets[i] + '/images/' + x for x in os.listdir(os.getcwd() + '/datasets/' + datasets[i] + '/images/')]
        Y += [os.getcwd() + '/datasets/' + datasets[i] + '/annotations/' + y for y in os.listdir(os.getcwd() + '/datasets/' + datasets[i] + '/annotations/')]
        pass
    
    # sort the file lists
    X.sort() 
    Y.sort()
    
    # make sure that the filenames match between the images (.jpg .png) and the annotations (.xml)
    assert([x.split('/')[-1][:-4] for x in X] == [y.split('/')[-1][:-4] for y in Y])
    
    return X, Y
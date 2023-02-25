import os
import glob
import random
import numpy as np

import torch


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def label_dict(label_path):
    """ Create a dictionary to store hero names corresponding to id classes 
    Args:
        label_path (str) -- A path of a file that contains all names of heros
    Returns:
        name_to_id (dict) -- {name_hero : id_class}
        id_to_name (dicy) -- {id_class : name_hero}
    """

    # Load hero names
    with open(label_path, 'r') as file:
        hero_names = file.readlines()
        # removes newline character
        hero_names = [hero_name.rstrip() for hero_name in hero_names]
    indexes = list(range(len(hero_names)))
    name_to_id = dict(zip(hero_names, indexes))
    id_to_name = dict(zip(indexes, hero_names))

    return name_to_id, id_to_name


def label_img(list_img_name, label_path):
    """ Funtion for labeling the image.
    Args:
        list_img_name (list) -- A list of image names will be labeled
        label_path (str) -- A path of a file that contains all names of heros
    Return:  
        labeled_data (dict) -- A dictionary stores labeled datas {image_name : id_class}
    """

    # Declare a dictionary stores labeled datas
    labeled_data = {}
    name_to_id, id_to_name = label_dict(label_path)

    # Load hero names
    with open(label_path, 'r') as file:
        hero_names = file.readlines()
        # removes newline character
        hero_names = [hero_name.rstrip() for hero_name in hero_names]

    for img_name in list_img_name:
        first_img_name = img_name.split('_')[0]
        for hero_name in hero_names:
            first_hero_name = hero_name.split('_')[0]
            if first_img_name == first_hero_name:
                labeled_data[img_name] = name_to_id[hero_name]
            elif first_img_name == 'Dr':
                labeled_data[img_name] = name_to_id['Dr._Mundo']

    return labeled_data


def split_data(data_path, label_path, validation_split = 0.2):
    """ Split data to a train and a test dataset 
    Args:
        data_path (str) -- A path of folder leads to all image paths
        validation_split (float) -- A ratio of the validation dataset
    Returns:
        train_data -- A list stores training dataset
        val_data -- A list stores validation dataset
    """

    # Make sure all hero names be in the train dataset
    occurrences = 1
    train_data = []
    val_data = []

    # Load hero names
    with open(label_path, 'r') as file:
        hero_names = file.readlines()
        # removes newline character
        hero_names = [hero_name.rstrip() for hero_name in hero_names]

    # Load image names
    list_img_name = os.listdir(data_path)

    # Label images
    labeled_img = label_img(list_img_name, label_path)
    num_images = len(labeled_img)

    # Make sure each class is included in the training dataset
    for idx in set(labeled_img.values()):
        for key, value in labeled_img.items():
            if value == idx:
                train_data.append(key)
                del labeled_img[key]
                break
                
    # Shuffle the rest of the dict and split it
    img_names = list(labeled_img.keys())
    random.shuffle(img_names)

    val_data = img_names[:int(num_images*validation_split)]
    train_data.extend([img_name for img_name in img_names[int(num_images*validation_split):]])
    
    return train_data, val_data
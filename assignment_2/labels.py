from collections import namedtuple
import numpy as np

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [
    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
    'id'          , # original Carla simulator id
    'trainId'     , # Unique Id of a label
    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.

labels = [
    #       name                       id     trainId   color
    Label(  'unlabeled'            ,   0 ,    255 , (  0,  0,  0) ),
    Label(  'building'             ,   1 ,      0 , ( 70, 70, 70) ),
    Label(  'fence'                ,   2 ,      1 , (100, 40, 40) ),
    Label(  'other'                ,   3 ,    255 , (  0,  0,  0) ),
    Label(  'pedestrian'           ,   4 ,      2 , (220, 20, 60) ),
    Label(  'pole'                 ,   5 ,      3 , (153,153,153) ),
    Label(  'road line'            ,   6 ,      4 , (157,234, 50) ),
    Label(  'road'                 ,   7 ,      5 , (128, 64,128) ),
    Label(  'sidewalk'             ,   8 ,      6 , (244, 35,232) ),
    Label(  'vegetation'           ,   9 ,      7 , (107,142, 35) ),
    Label(  'vehicles'             ,  10 ,      8 , (  0,  0,142) ),
    Label(  'wall'                 ,  11 ,      9 , (102,102,156) ),
    Label(  'traffic sign'         ,  12 ,     10 , (220,220,  0) ),
    Label(  'sky'                  ,  13 ,     11 , ( 70,130,180) ),
    Label(  'ground'               ,  14 ,    255 , (  0,  0,  0) ),
    Label(  'bridge'               ,  15 ,    255 , (  0,  0,  0) ),
    Label(  'rail_track'           ,  16 ,    255 , (  0,  0,  0) ),
    Label(  'guard rail'           ,  17 ,     12 ,  (180,165,180)),
    Label(  'traffic light'        ,  18 ,     13 , (250,170, 30) ),
    Label(  'static'               ,  19 ,    255 , (  0,  0,  0) ),
    Label(  'dynamic'              ,  20 ,    255 , (  0,  0,  0) ),
    Label(  'water'                ,  21 ,    255 , (  0,  0,  0) ),
    Label(  'terrain'              ,  22 ,     14 , (145,170,100) ),
]

def label_to_color(seg, num_classes):
    rgb_seg = np.zeros((seg.shape[0], seg.shape[1], 3))
    for i in range(num_classes):
        color = labels[int(np.where(np.array([labels[i].trainId for i in range(len(labels))]) == i)[0])].color

        rgb_seg = np.where(np.expand_dims(seg[:,:], axis=-1) == i,
                           np.ones_like(rgb_seg) * np.expand_dims(np.expand_dims(np.array(color), axis=0), axis=0),
                           rgb_seg)
    return rgb_seg.astype(np.uint8)














 # THIS IS ONLY FOR THE ASSIGNMENT SHEET!
'''
#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .

    'trainId'     , # Unique Id of a label

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.

labels = [
    #       name                       trainId   color
    Label(  'unlabeled'            ,       255 , (  0,  0,  0) ),
    Label(  'building'             ,         0 , ( 70, 70, 70) ),
    Label(  'fence'                ,         1 , (100, 40, 40) ),
    Label(  'other'                ,         2 , ( 55, 90, 80) ),
    Label(  'pedestrian'           ,         3 , (220, 20, 60) ),
    Label(  'pole'                 ,         4 , (153,153,153) ),
    Label(  'road line'            ,         5 , (157,234, 50) ),
    Label(  'road'                 ,         6 , (128, 64,128) ),
    Label(  'sidewalk'             ,         7 , (244, 35,232) ),
    Label(  'vegetation'           ,         8 , (107,142, 35) ),
    Label(  'vehicles'             ,         9 , (  0,  0,142) ),
    Label(  'wall'                 ,        10 , (102,102,156) ),
    Label(  'traffic sign'         ,        11 , (220,220,  0) ),
    Label(  'sky'                  ,        12 , ( 70,130,180) ),
    Label(  'guard rail'           ,        13 ,  (180,165,180) ),
    Label(  'traffic light'        ,        14 , (250,170, 30) ),
    Label(  'terrain'              ,        15 , (145,170,100) ),
]

'''
'''
labels = [
    #       name                       id     trainId   color
    Label(  'unlabeled'            ,   0 ,    255 , (  0,  0,  0) ),
    Label(  'building'             ,   1 ,      0 , ( 70, 70, 70) ),
    Label(  'fence'                ,   2 ,      1 , (100, 40, 40) ),
    Label(  'other'                ,   3 ,    255 , (  0,  0,  0) ),
    Label(  'pedestrian'           ,   4 ,      2 , (220, 20, 60) ),
    Label(  'pole'                 ,   5 ,      3 , (153,153,153) ),
    Label(  'road line'            ,   6 ,      4 , (157,234, 50) ),
    Label(  'road'                 ,   7 ,      5 , (128, 64,128) ),
    Label(  'sidewalk'             ,   8 ,      6 , (244, 35,232) ),
    Label(  'vegetation'           ,   9 ,      7 , (107,142, 35) ),
    Label(  'vehicles'             ,  10 ,      8 , (  0,  0,142) ),
    Label(  'wall'                 ,  11 ,      9 , (102,102,156) ),
    Label(  'traffic sign'         ,  12 ,     10 , (220,220,  0) ),
    Label(  'sky'                  ,  13 ,     11 , ( 70,130,180) ),
    Label(  'ground'               ,  14 ,    255 , (  0,  0,  0) ),
    Label(  'bridge'               ,  15 ,    255 , (  0,  0,  0) ),
    Label(  'rail_track'           ,  16 ,    255 , (  0,  0,  0) ),
    Label(  'guard rail'           ,  17 ,     12 ,  (180,165,180)),
    Label(  'traffic light'        ,  18 ,     13 , (250,170, 30) ),
    Label(  'static'               ,  19 ,    255 , (  0,  0,  0) ),
    Label(  'dynamic'              ,  20 ,    255 , (  0,  0,  0) ),
    Label(  'water'                ,  21 ,    255 , (  0,  0,  0) ),
    Label(  'terrain'              ,  22 ,     14 , (145,170,100) ),
]
'''
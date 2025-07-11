# All coco categories, together with their nice-looking visualization colors
# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json
COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {
        "color": [175, 116, 175],
        "isthing": 1,
        "id": 14,
        "name": "parking meter",
    },
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    {
        "color": [255, 208, 186],
        "isthing": 1,
        "id": 43,
        "name": "tennis racket",
    },
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
    {"color": [255, 255, 128], "isthing": 0, "id": 92, "name": "banner"},
    {"color": [147, 211, 203], "isthing": 0, "id": 93, "name": "blanket"},
    {"color": [150, 100, 100], "isthing": 0, "id": 95, "name": "bridge"},
    {"color": [168, 171, 172], "isthing": 0, "id": 100, "name": "cardboard"},
    {"color": [146, 112, 198], "isthing": 0, "id": 107, "name": "counter"},
    {"color": [210, 170, 100], "isthing": 0, "id": 109, "name": "curtain"},
    {"color": [92, 136, 89], "isthing": 0, "id": 112, "name": "door-stuff"},
    {"color": [218, 88, 184], "isthing": 0, "id": 118, "name": "floor-wood"},
    {"color": [241, 129, 0], "isthing": 0, "id": 119, "name": "flower"},
    {"color": [217, 17, 255], "isthing": 0, "id": 122, "name": "fruit"},
    {"color": [124, 74, 181], "isthing": 0, "id": 125, "name": "gravel"},
    {"color": [70, 70, 70], "isthing": 0, "id": 128, "name": "house"},
    {"color": [255, 228, 255], "isthing": 0, "id": 130, "name": "light"},
    {"color": [154, 208, 0], "isthing": 0, "id": 133, "name": "mirror-stuff"},
    {"color": [193, 0, 92], "isthing": 0, "id": 138, "name": "net"},
    {"color": [76, 91, 113], "isthing": 0, "id": 141, "name": "pillow"},
    {"color": [255, 180, 195], "isthing": 0, "id": 144, "name": "platform"},
    {
        "color": [106, 154, 176],
        "isthing": 0,
        "id": 145,
        "name": "playingfield",
    },
    {"color": [230, 150, 140], "isthing": 0, "id": 147, "name": "railroad"},
    {"color": [60, 143, 255], "isthing": 0, "id": 148, "name": "river"},
    {"color": [128, 64, 128], "isthing": 0, "id": 149, "name": "road"},
    {"color": [92, 82, 55], "isthing": 0, "id": 151, "name": "roof"},
    {"color": [254, 212, 124], "isthing": 0, "id": 154, "name": "sand"},
    {"color": [73, 77, 174], "isthing": 0, "id": 155, "name": "sea"},
    {"color": [255, 160, 98], "isthing": 0, "id": 156, "name": "shelf"},
    {"color": [255, 255, 255], "isthing": 0, "id": 159, "name": "snow"},
    {"color": [104, 84, 109], "isthing": 0, "id": 161, "name": "stairs"},
    {"color": [169, 164, 131], "isthing": 0, "id": 166, "name": "tent"},
    {"color": [225, 199, 255], "isthing": 0, "id": 168, "name": "towel"},
    {"color": [137, 54, 74], "isthing": 0, "id": 171, "name": "wall-brick"},
    {"color": [135, 158, 223], "isthing": 0, "id": 175, "name": "wall-stone"},
    {"color": [7, 246, 231], "isthing": 0, "id": 176, "name": "wall-tile"},
    {"color": [107, 255, 200], "isthing": 0, "id": 177, "name": "wall-wood"},
    {"color": [58, 41, 149], "isthing": 0, "id": 178, "name": "water-other"},
    {
        "color": [183, 121, 142],
        "isthing": 0,
        "id": 180,
        "name": "window-blind",
    },
    {"color": [255, 73, 97], "isthing": 0, "id": 181, "name": "window-other"},
    {"color": [107, 142, 35], "isthing": 0, "id": 184, "name": "tree-merged"},
    {
        "color": [190, 153, 153],
        "isthing": 0,
        "id": 185,
        "name": "fence-merged",
    },
    {
        "color": [146, 139, 141],
        "isthing": 0,
        "id": 186,
        "name": "ceiling-merged",
    },
    {
        "color": [70, 130, 180],
        "isthing": 0,
        "id": 187,
        "name": "sky-other-merged",
    },
    {
        "color": [134, 199, 156],
        "isthing": 0,
        "id": 188,
        "name": "cabinet-merged",
    },
    {
        "color": [209, 226, 140],
        "isthing": 0,
        "id": 189,
        "name": "table-merged",
    },
    {
        "color": [96, 36, 108],
        "isthing": 0,
        "id": 190,
        "name": "floor-other-merged",
    },
    {
        "color": [96, 96, 96],
        "isthing": 0,
        "id": 191,
        "name": "pavement-merged",
    },
    {
        "color": [64, 170, 64],
        "isthing": 0,
        "id": 192,
        "name": "mountain-merged",
    },
    {
        "color": [152, 251, 152],
        "isthing": 0,
        "id": 193,
        "name": "grass-merged",
    },
    {"color": [208, 229, 228], "isthing": 0, "id": 194, "name": "dirt-merged"},
    {
        "color": [206, 186, 171],
        "isthing": 0,
        "id": 195,
        "name": "paper-merged",
    },
    {
        "color": [152, 161, 64],
        "isthing": 0,
        "id": 196,
        "name": "food-other-merged",
    },
    {
        "color": [116, 112, 0],
        "isthing": 0,
        "id": 197,
        "name": "building-other-merged",
    },
    {"color": [0, 114, 143], "isthing": 0, "id": 198, "name": "rock-merged"},
    {
        "color": [102, 102, 156],
        "isthing": 0,
        "id": 199,
        "name": "wall-other-merged",
    },
    {"color": [250, 141, 255], "isthing": 0, "id": 200, "name": "rug-merged"},
]

# Novel COCO categories
COCO_NOVEL_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
]


CAMO_CATEGORIES = [
    {'name': 'Frog',            'id': 0,   'isthing': 1,    'color': [220, 20, 60]},
    {'name': 'Crab',            'id': 1,   'isthing': 1,    'color': [119, 11, 32]},
    {'name': 'Dolphin',         'id': 2,   'isthing': 1,    'color': [0, 0, 142]},
    {'name': 'Fish',            'id': 3,   'isthing': 1,    'color': [0, 0, 230]},
    {'name': 'Octopus',         'id': 4,   'isthing': 1,    'color': [106, 0, 228]},
    {'name': 'Seahorse',        'id': 5,   'isthing': 1,    'color': [0, 60, 100]},
    {'name': 'Shrimp',          'id': 6,   'isthing': 1,    'color': [0, 80, 100]},
    {'name': 'Starfish',        'id': 7,   'isthing': 1,    'color': [0, 0, 70]},
    {'name': 'Scorpion',        'id': 8,   'isthing': 1,    'color': [0, 0, 192]},
    {'name': 'Spider',          'id': 9,   'isthing': 1,    'color': [250, 170, 30]},
    {'name': 'Bird',            'id': 10,  'isthing': 1,    'color': [100, 170, 30]},
    {'name': 'Penguin',         'id': 11,  'isthing': 1,    'color': [220, 220, 0]},
    {'name': 'Snail',           'id': 12,  'isthing': 1,    'color': [250, 0, 30]},
    {'name': 'Insect',                             'id': 13,  'isthing': 1,    'color': [165, 42, 42]},
    {'name': 'Moth',            'id': 14,  'isthing': 1,    'color': [255, 77, 255]},
    {'name': 'Bat',             'id': 15,  'isthing': 1,    'color': [0, 226, 252]},
    {'name': 'Bear',            'id': 16,  'isthing': 1,    'color': [182, 182, 255]},
    {'name': 'Camel',           'id': 17,  'isthing': 1,    'color': [0, 82, 0]},
    {'name': 'Cat',             'id': 18,  'isthing': 1,    'color': [120, 166, 157]},
    {'name': 'Deer',            'id': 19,  'isthing': 1,    'color': [110, 76, 0]},
    {'name': 'Dog',             'id': 20,  'isthing': 1,    'color': [174, 57, 255]},
    {'name': 'Elephant',        'id': 21,  'isthing': 1,    'color': [199, 100, 0]},
    {'name': 'Fox',             'id': 22,  'isthing': 1,    'color': [72, 0, 118]},
    {'name': 'Giraffe',         'id': 23,  'isthing': 1,    'color': [255, 179, 240]},
    {'name': 'Goat',            'id': 24,  'isthing': 1,    'color': [0, 125, 92]},
    {'name': 'Hedgehog',        'id': 25,  'isthing': 1,    'color': [209, 0, 151]},
    {'name': 'Horse',           'id': 26,  'isthing': 1,    'color': [188, 208, 182]},
    {'name': 'Kangaroo',        'id': 27,  'isthing': 1,    'color': [0, 220, 176]},
    {'name': 'Leopard',         'id': 28,  'isthing': 1,    'color': [255, 99, 164]},
    {'name': 'Lion',            'id': 29,  'isthing': 1,    'color': [92, 0, 73]},
    {'name': 'Monkey',          'id': 30,  'isthing': 1,    'color': [133, 129, 255]},
    {'name': 'Mouse',           'id': 31,  'isthing': 1,    'color': [78, 180, 255]},
    {'name': 'Pig',             'id': 32,  'isthing': 1,    'color': [0, 228, 0]},
    {'name': 'Rabbit',          'id': 33,  'isthing': 1,    'color': [174, 255, 243]},
    {'name': 'Rhino',           'id': 34,  'isthing': 1,    'color': [45, 89, 255]},
    {'name': 'Sea_Lion',        'id': 35,  'isthing': 1,    'color': [134, 134, 103]},
    {'name': 'Sloth',           'id': 36,  'isthing': 1,    'color': [145, 148, 174]},
    {'name': 'Squirrel',        'id': 37,  'isthing': 1,    'color': [197, 226, 255]},
    {'name': 'Tiger',           'id': 38,  'isthing': 1,    'color': [171, 134, 1]},
    {'name': 'Weasel',          'id': 39,  'isthing': 1,    'color': [109, 63, 54]},
    {'name': 'Body_Painting',   'id': 40,  'isthing': 1,    'color': [207, 138, 255]},
    {'name': 'Solider',         'id': 41,  'isthing': 1,    'color': [151, 0, 95]},
    {'name': 'Crocodile',       'id': 42,  'isthing': 1,    'color': [9, 80, 61]},
    {'name': 'Lizard',          'id': 43,  'isthing': 1,    'color': [84, 105, 51]},
    {'name': 'Snake',           'id': 44,  'isthing': 1,    'color': [74, 65, 105]},
    {'name': 'Turtle',          'id': 45,  'isthing': 1,    'color': [166, 196, 102]},
    {'name': 'Worm',            'id': 46,  'isthing': 1,    'color': [208, 195, 210]},
]

# CAMO_NOVEL_CATEGORIES = {
#     1: ['Shrimp', 'Crocodile', 'Snail', 'Bat', 'Mouse', 'Hedgehog', 'Tiger', 'Squirrel', 'Scorpion', 'Bear'], # 'Elephant'
#     2: ['Turtle', 'Worm', 'Goat', 'Deer', 'Lion', 'Sloth', 'Rhino', 'Kangaroo', 'Monkey', 'Horse'], #
#     3: ['Octopus', 'Starfish', 'Seahorse', 'Crab', 'Fish', 'Frog', 'Snake', 'Lizard', 'Giraffe', 'Leopard', 'Cat', 'Sea_Lion', 'Fox', 'Weasel', 'Rabbit', 'Dog', 'Insect', 'Moth', 'Bird', 'Body_Painting', 'Solider', 'Spider', 'Penguin'],  #, 'Dolphin'
# }
# CAMO_NOVEL_CATEGORIES = {
#     1: ["Shrimp", "Crocodile", "Snail", "Elephant", "Goat",
#         "Lion", "Bat", "Camel", "Mouse", "Hedgehog",
#         "Tiger", "Pig", "Squirrel", "Sloth", "Scorpion",
#         "Bear"], # 16

#     2: ["Crab", "Dolphin", "Turtle", "Worm", "Kangaroo",
#         "Giraffe", "Leopard", "Monkey", "Deer", "Sea_Lion",
#         "Fox", "Weasel", "Rabbit", "Dog", "Horse",
#         "Rhino"] , # 16

#     3: ["Octopus", "Starfish", "Seahorse", "Fish", "Frog",
#         "Snake", "Lizard", "Cat", "Insect", "Moth",
#         "Bird", "Penguin", "Body_Painting", "Solider", "Spider"],    # 15
# }

#CAMO_NOVEL_CATEGORIES = {
#     1: ["Shrimp", "Crab", "Dolphin", "Crocodile", "Snake", "Turtle", "Worm", "Snail", "Kangaroo", "Elephant", "Giraffe", "Goat",
#         "Leopard", "Monkey", "Deer", "Sea_Lion", "Lion", "Bat", "Fox", "Camel", "Weasel", "Rabbit", "Dog", "Horse", "Mouse", "Hedgehog",
#         "Rhino", "Tiger", "Pig", "Squirrel", "Sloth", "Scorpion", "Bear"], # 33
#     2: ["Octopus", "Starfish", "Seahorse", "Fish", "Frog", "Lizard", "Cat", "Moth", "Penguin", "Spider"], # 10
#     3: ["Insect", "Bird", "Body_Painting", "Solider"] # 4
#}

# split4 cach chia 1
#CAMO_NOVEL_CATEGORIES = {
#    1: [        "Shrimp",        "Crocodile",        "Snail",        "Elephant",        "Goat",        "Lion",        "Bat",        "Camel",        "Mouse",
#        "Hedgehog",        "Tiger",        "Pig",       "Squirrel",        "Sloth",        "Scorpion",        "Bear",    ],  # 16
#    2: [        "Crab",        "Dolphin",        "Turtle",       "Worm",        "Kangaroo",        "Giraffe",        "Leopard",        "Monkey",        "Deer",
#        "Sea_Lion",        "Fox",        "Weasel",        "Rabbit",        "Dog",        "Horse",        "Bird",        "Rhino",        "Body_Painting",    ],  # 18
#    3: [        "Octopus",        "Starfish",        "Seahorse",        "Fish",        "Frog",        "Snake",        "Lizard",        "Cat",        "Insect",
#        "Moth",        "Penguin",        "Solider",        "Spider",    ],  # 13
#}

## split5 cach chia 2
#CAMO_NOVEL_CATEGORIES = {
#    1: ["Shrimp", "Crab", "Dolphin", "Crocodile", "Snake", "Turtle", "Worm", "Snail", "Kangaroo", "Elephant", "Giraffe", "Goat", "Leopard", "Monkey", "Deer", "Sea_Lion", "Lion", "Bat", "Fox", "Camel", "Weasel", "Rabbit", "Dog", #"Horse", "Mouse", "Hedgehog", "Rhino", "Tiger", "Pig", "Squirrel", "Sloth", "Scorpion", "Bear"], #33
#    2: ["Octopus", "Starfish", "Seahorse", "Fish", "Frog", "Lizard", "Cat", "Moth", "Penguin", "Spider"], #10
#    3: ["Insect", "Bird", "Body_Painting", "Solider"] #4
#}

# split6 noset
CAMO_NOVEL_CATEGORIES = {
    1: ["Shrimp", "Crab", "Dolphin", "Crocodile", "Snake", "Turtle", "Worm", "Snail", "Kangaroo", "Elephant", "Giraffe", "Goat", "Leopard", "Monkey", "Deer", "Sea_Lion", "Lion", "Bat", "Fox", "Camel", "Weasel", "Rabbit", "Dog", "Horse", "Mouse", "Hedgehog", "Rhino", "Tiger", "Pig", "Squirrel", "Sloth", "Scorpion", "Bear", "Octopus", "Starfish", "Seahorse", "Fish", "Frog", "Lizard", "Cat", "Moth", "Penguin", "Spider", "Insect", "Bird", "Body_Painting", "Solider"],
    2: [],
    3: [],
    25: ["Shrimp", "Crab", "Dolphin", "Crocodile", "Snake", "Turtle", "Worm", "Snail", "Kangaroo", "Elephant", "Giraffe", "Goat", "Leopard", "Monkey", "Deer", "Sea_Lion", "Lion", "Bat", "Fox", "Camel", "Weasel", "Rabbit", "Dog", "Horse", "Mouse", "Hedgehog", "Rhino", "Tiger", "Pig", "Squirrel", "Sloth", "Scorpion", "Bear", "Octopus", "Starfish", "Seahorse", "Fish", "Frog", "Lizard", "Cat", "Moth", "Penguin", "Spider", "Insect", "Bird", "Body_Painting", "Solider"],
}


# PASCAL VOC categories
PASCAL_VOC_ALL_CATEGORIES = {
    1: [
        "aeroplane",
        "bicycle",
        "boat",
        "bottle",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
        "bird",
        "bus",
        "cow",
        "motorbike",
        "sofa",
    ],
    2: [
        "bicycle",
        "bird",
        "boat",
        "bus",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
        "aeroplane",
        "bottle",
        "cow",
        "horse",
        "sofa",
    ],
    3: [
        "aeroplane",
        "bicycle",
        "bird",
        "bottle",
        "bus",
        "car",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "train",
        "tvmonitor",
        "boat",
        "cat",
        "motorbike",
        "sheep",
        "sofa",
    ],
}

PASCAL_VOC_NOVEL_CATEGORIES = {
    1: ["bird", "bus", "cow", "motorbike", "sofa"],
    2: ["aeroplane", "bottle", "cow", "horse", "sofa"],
    3: ["boat", "cat", "motorbike", "sheep", "sofa"],
}

PASCAL_VOC_BASE_CATEGORIES = {
    1: [
        "aeroplane",
        "bicycle",
        "boat",
        "bottle",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
    ],
    2: [
        "bicycle",
        "bird",
        "boat",
        "bus",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
    ],
    3: [
        "aeroplane",
        "bicycle",
        "bird",
        "bottle",
        "bus",
        "car",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "train",
        "tvmonitor",
    ],
}


def _get_coco_instances_meta():
    thing_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 80, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_coco_fewshot_instances_meta():
    ret = _get_coco_instances_meta()
    novel_ids = [k["id"] for k in COCO_NOVEL_CATEGORIES if k["isthing"] == 1]
    novel_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(novel_ids)}
    novel_classes = [
        k["name"] for k in COCO_NOVEL_CATEGORIES if k["isthing"] == 1
    ]
    base_categories = [
        k
        for k in COCO_CATEGORIES
        if k["isthing"] == 1 and k["name"] not in novel_classes
    ]
    base_ids = [k["id"] for k in base_categories]
    base_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(base_ids)}
    base_classes = [k["name"] for k in base_categories]
    ret[
        "novel_dataset_id_to_contiguous_id"
    ] = novel_dataset_id_to_contiguous_id
    ret["novel_classes"] = novel_classes
    ret["base_dataset_id_to_contiguous_id"] = base_dataset_id_to_contiguous_id
    ret["base_classes"] = base_classes
    return ret


def _get_camo_instances_meta():
    # get ID COCO BASE
    split=0
    novel_categories = COCO_NOVEL_CATEGORIES#[split]
    novel_classes = [k["name"] for k in novel_categories if k["isthing"] == 1]
    base_categories = [k for k in COCO_CATEGORIES
                       if k["isthing"] == 1 and k["name"] not in novel_classes]
    base_ids = [k["id"] for k in base_categories]
    base_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(base_ids)}



    # register CAMO

    thing_ids = [k["id"] for k in CAMO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in CAMO_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 47, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in CAMO_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
        'base_dataset_id_to_contiguous_id': base_dataset_id_to_contiguous_id
    }
    return ret

def _get_camo_fewshot_instances_meta():
    name2id = {ins['name']:ins['id'] for ins in CAMO_CATEGORIES}

    all_ret = _get_camo_instances_meta()
    fewshot_ret = {}
    for split, novel_classes in CAMO_NOVEL_CATEGORIES.items():
        novel_ids = [name2id[name] for name in novel_classes]
        novel_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(novel_ids)}

        fewshot_ret[split] = dict(all_ret)
        fewshot_ret[split].update(
            {
            'novel_classes': novel_classes,
            'novel_ids': novel_ids,
            'novel_dataset_id_to_contiguous_id': novel_dataset_id_to_contiguous_id,
            'base_classes': [],
            'thing_classes': [],
            }
        )

    return fewshot_ret


def _get_lvis_instances_meta_v0_5():
    from .lvis_v0_5_categories import LVIS_CATEGORIES

    assert len(LVIS_CATEGORIES) == 1230
    cat_ids = [k["id"] for k in LVIS_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    lvis_categories = [
        k for k in sorted(LVIS_CATEGORIES, key=lambda x: x["id"])
    ]
    thing_classes = [k["synonyms"][0] for k in lvis_categories]
    meta = {"thing_classes": thing_classes}
    return meta


def _get_lvis_fewshot_instances_meta_v0_5():
    from .lvis_v0_5_categories import LVIS_CATEGORIES_NOVEL

    all_cats = _get_lvis_instances_meta_v0_5()["thing_classes"]
    lvis_categories_sub = [
        k for k in sorted(LVIS_CATEGORIES_NOVEL, key=lambda x: x["id"])
    ]
    sub_cats = [k["synonyms"][0] for k in lvis_categories_sub]
    mapping = {all_cats.index(c): i for i, c in enumerate(sub_cats)}
    meta = {"thing_classes": sub_cats, "class_mapping": mapping}

    return meta


def _get_pascal_voc_fewshot_instances_meta():
    ret = {
        "thing_classes": PASCAL_VOC_ALL_CATEGORIES,
        "novel_classes": PASCAL_VOC_NOVEL_CATEGORIES,
        "base_classes": PASCAL_VOC_BASE_CATEGORIES,
    }
    return ret


#def _get_builtin_metadata(dataset_name):
#    if dataset_name == "coco":
#        return _get_coco_instances_meta()
#    elif dataset_name == "coco_fewshot":
#        return _get_coco_fewshot_instances_meta()
#    elif dataset_name == "lvis_v0.5":
#        return _get_lvis_instances_meta_v0_5()
#    elif dataset_name == "lvis_v0.5_fewshot":
#        return _get_lvis_fewshot_instances_meta_v0_5()
#    elif dataset_name == "pascal_voc_fewshot":
#        return _get_pascal_voc_fewshot_instances_meta()
#    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))

def _get_builtin_metadata(dataset_name, split=0):
    if dataset_name == "coco":
        return _get_coco_instances_meta()
    elif dataset_name == "coco_fewshot":
        return _get_coco_fewshot_instances_meta()
    elif dataset_name == "lvis_v0.5":
        return _get_lvis_instances_meta_v0_5()
    elif dataset_name == "lvis_v0.5_fewshot":
        return _get_lvis_fewshot_instances_meta_v0_5()
    if dataset_name == "coco_base_support":
        return _get_coco_base_support_instances_meta()
    if dataset_name == "coco_panoptic_separated":
        return _get_coco_panoptic_separated_meta()
    elif dataset_name == "coco_person":
        return {
            "thing_classes": ["person"],
            "keypoint_names": COCO_PERSON_KEYPOINT_NAMES,
            "keypoint_flip_map": COCO_PERSON_KEYPOINT_FLIP_MAP,
            "keypoint_connection_rules": KEYPOINT_CONNECTION_RULES,
        }
    elif dataset_name == "cityscapes":
        # fmt: off
        CITYSCAPES_THING_CLASSES = [
            "person", "rider", "car", "truck",
            "bus", "train", "motorcycle", "bicycle",
        ]
        CITYSCAPES_STUFF_CLASSES = [
            "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
            "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", "bicycle", "license plate",
        ]
        # fmt: on
        return {
            "thing_classes": CITYSCAPES_THING_CLASSES,
            "stuff_classes": CITYSCAPES_STUFF_CLASSES,
        }
    elif dataset_name == "camo_fewshot":
        return _get_camo_fewshot_instances_meta()
    elif dataset_name == "camo_fewshot_v4":
        return _get_camo_fewshot_instances_meta()
    elif dataset_name == "camo_fewshot_v5":
        return _get_camo_fewshot_instances_meta()
    elif dataset_name == "camo_fewshot_v4_only":
        return _get_camo_fewshot_instances_meta()
    elif dataset_name == "camo_fewshot_v5_only":
        return _get_camo_fewshot_instances_meta()

    elif dataset_name == "camo_{}_fewshot".format("blendeddiff_v0"):
        return _get_camo_fewshot_instances_meta()
    elif dataset_name == "camo_{}_fewshot".format("diffinpainting_v0"):
        return _get_camo_fewshot_instances_meta()
    elif dataset_name == "camo_{}_fewshot".format("gligen_v0"):
        return _get_camo_fewshot_instances_meta()

    elif dataset_name == "camo_{}_fewshot".format("blendeddiff_v1_histogramfiltering"):
        return _get_camo_fewshot_instances_meta()
    elif dataset_name == "camo_{}_fewshot".format("diffinpainting_v1_histogramfiltering"):
        return _get_camo_fewshot_instances_meta()
    elif dataset_name == "camo_{}_fewshot".format("gligen_v1_histogramfiltering"):
        return _get_camo_fewshot_instances_meta()

    elif dataset_name == "camo_{}_fewshot".format("blendeddiff_v2_histogrammatching"):
        return _get_camo_fewshot_instances_meta()
    elif dataset_name == "camo_{}_fewshot".format("diffinpainting_v2_histogrammatching"):
        return _get_camo_fewshot_instances_meta()
    elif dataset_name == "camo_{}_fewshot".format("gligen_v2_histogrammatching"):
        return _get_camo_fewshot_instances_meta()

    elif dataset_name == "camo_{}_fewshot".format("blendeddiff_v3_histogrammatching_1real_4syn"):
        return _get_camo_fewshot_instances_meta()
    elif dataset_name == "camo_{}_fewshot".format("diffinpainting_v3_histogrammatching_1real_4syn"):
        return _get_camo_fewshot_instances_meta()
    elif dataset_name == "camo_{}_fewshot".format("gligen_v3_histogrammatching_1real_4syn"):
        return _get_camo_fewshot_instances_meta()

    elif dataset_name == "camo_{}_fewshot".format("blendeddiff_v4_histogrammatching_1real_4syn_noleak"):
        return _get_camo_fewshot_instances_meta()
    elif dataset_name == "camo_{}_fewshot".format("diffinpainting_v4_histogrammatching_1real_4syn_noleak"):
        return _get_camo_fewshot_instances_meta()
    elif dataset_name == "camo_{}_fewshot".format("gligen_v4_histogrammatching_1real_4syn_noleak"):
        return _get_camo_fewshot_instances_meta()

    elif dataset_name == "camo_{}_fewshot".format("datazoo_v5_histogrammatching_1real_4syn_noleak"):
        return _get_camo_fewshot_instances_meta()
    elif dataset_name == "camo_{}_fewshot".format("datazoo_v5_1real_4syn_noleak"):
        return _get_camo_fewshot_instances_meta()

    elif dataset_name == "camo_{}_fewshot".format("blendeddiff_v5_background_1real_4syn_noleak"):
        return _get_camo_fewshot_instances_meta()
    elif dataset_name == "camo_{}_fewshot".format("diffinpainting_v5_background_1real_4syn_noleak"):
        return _get_camo_fewshot_instances_meta()
    elif dataset_name == "camo_{}_fewshot".format("gligen_v5_background_1real_4syn_noleak"):
        return _get_camo_fewshot_instances_meta()

    elif dataset_name == "camo_{}_fewshot".format("blendeddiff_v6_background_1real_1syn_noleak"):
        return _get_camo_fewshot_instances_meta()
    elif dataset_name == "camo_{}_fewshot".format("diffinpainting_v6_background_1real_1syn_noleak"):
        return _get_camo_fewshot_instances_meta()
    elif dataset_name == "camo_{}_fewshot".format("gligen_v6_background_1real_1syn_noleak"):
        return _get_camo_fewshot_instances_meta()


    elif dataset_name == "camo_{}_fewshot".format("blendeddiff_v7_background_1real_1syn_noleak_noinstances"):
        return _get_camo_fewshot_instances_meta()


    elif dataset_name == "pascal_voc_fewshot":
        return _get_pascal_voc_fewshot_instances_meta()
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
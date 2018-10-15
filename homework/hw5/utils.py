from os import listdir
from os.path import join
import random

def generate_train_img_names(root_path, out_path):
    """Get path and label of training images.
    Produce a txt file where each line contains the path and label separated by sapce,
    such as 'data/tiny-imagenet-200/train/n01443537/images/n01443537_0.JPEG n01443537'
    Args:
        root_path(str): similar to "data/tiny-imagenet-200/train/"
        out_path(str): txt file to store the results
    """

    with open(out_path, "w") as file:
        for class_name in listdir(root_path):
            if class_name[0] == "n":
                full_path = root_path + class_name + "/images"
                for image in listdir(full_path):
                    file.write(join(full_path, image) + " " + class_name + "\n")

def generate_val_img_names(root_path, out_path):
    """Get path and label of validate images.
    Produce a txt file where each line contains the path and label separated by sapce,
    such as 'data/tiny-imagenet-200/val/images/val_0.JPEG n03444034'
    Args:
        root_path(str): similar to "data/tiny-imagenet-200/val/"
        out_path(str): txt file to store the results
    """

    out = open(out_path, "w")
    
    with open(root_path + "val_annotations.txt", "r") as file:
        for line in file:
            image_name, label = line.split()[0], line.split()[1]
            out.write(root_path + "images/" + image_name + " " + label + "\n")
    
    out.close()

# Do online sampling instead
# def generate_triplets(num_epochs, out_path):
#     """Generate triplets. Each triplet set contains three integers [i, j, k] 
#     corresponding to the index of [query, positive, negative] images in 
#     the train_list file.
#     Args:
#         num_epochs(int): epochs for training,
#                 for every epoch, generate 100000 triplets
#         out_path(str): txt file to store the results
#     """

#     out = open(out_path, "w")

#     for _ in range(num_epochs):
#         # 200 classes, each has 500 images, 100000 in total
#         count = 0  # from 0 to 199
#         step = 500
#         start = 0
#         end = start + step - 1
#         total = 100000
#         for i in range(total):
#             # Identify the current class
#             if i != 0 and i % step == 0:
#                 count += 1
#                 start = count * step
#                 end = start + step - 1

#             # Query image
#             query_idx = i

#             # Sample positive image within the same class
#             positive_idx = random.randint(start, end)
#             while positive_idx == query_idx:
#                 positive_idx = random.randint(start, end)
            
#             # Sample positive image in other class
#             range1 = list(range(0, start))
#             range2 = list(range(end + 1, total))
#             negative_idx = random.sample(range1 + range2, 1)[0]

#             out.write("{} {} {}\n".format(query_idx, positive_idx, negative_idx))
    
#     out.close()


    



import numpy as np
from helperFunctions import getUCF101

def get_most_confused_classes(confusion_matrix, class_list, k=10):
    """Get the top k most confused classes from the confusion matrix
    Args:
        confusion_matrix(num_classes x num_classes)
        k: the number of the top classes
        class_list: a list of the names of all the classes
    """
    
    # Exclude the diagonal elements.
    copied_matrix = np.copy(confusion_matrix)
    np.fill_diagonal(copied_matrix, 0)
    
    # Return the k largest row and col indices 
    flat = copied_matrix.flatten()
    indices = np.argpartition(flat, -k)[-k:]
    indices = indices[np.argsort(-flat[indices])]
    rows, cols = np.unravel_index(indices, confusion_matrix.shape)
    
    # Obtain the class name
    top_classes = []
    for row, col in zip(rows, cols):
        top_classes.append((class_list[row], class_list[col]))
    
    return top_classes

def combine_model_analysis(pred_path1, pred_path2, class_list, test, num_classes):
    prediction1 = np.load(pred_path1)
    prediction2 = np.load(pred_path2)

    combined_pred = (prediction1 + prediction2) / 2
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)

    acc_top1 = 0.0
    acc_top5 = 0.0
    acc_top10 = 0.0
    random_indices = np.random.permutation(len(test[0]))

    for i in range(len(test[0])):
        index = random_indices[i]

        label = test[1][index]
        curt_pred = combined_pred[index]
        argsort_pred = np.argsort(-curt_pred)[0:10]
        confusion_matrix[label, argsort_pred[0]] += 1

        if label == argsort_pred[0]:
            acc_top1 += 1.0
        if np.any(argsort_pred[0:5] == label):
            acc_top5 += 1.0
        if np.any(argsort_pred[:]==label):
            acc_top10 += 1.0
        
        print('i:%d (%f,%f,%f)' 
          % (i, acc_top1/(i+1), acc_top5/(i+1), acc_top10/(i+1)))
    
    number_of_examples = np.sum(confusion_matrix,axis=1)   # num examples of videos classfied into each class
    for i in range(num_classes):
        confusion_matrix[i,:] = confusion_matrix[i,:]/np.sum(confusion_matrix[i,:])

    results = np.diag(confusion_matrix)
    indices = np.argsort(results)

    sorted_list = np.asarray(class_list)
    sorted_list = sorted_list[indices]
    sorted_results = results[indices]

    for i in range(num_classes):
        # 1. name of a class, 
        # 2. percent of samples of this class being classfied correctly, 
        # 3. total num of samples classfied as this class
        print(sorted_list[i],sorted_results[i],number_of_examples[indices[i]])

    np.save('combined_confusion_matrix.npy', confusion_matrix)

if __name__ == '__main__':
    num_classes = 101

    data_directory = '/projects/training/bauh/AR/'
    class_list, train, test = getUCF101(base_directory = data_directory)

    # # Single frame model
    # confusion_matrix_single = np.load("single_frame_confusion_matrix.npy")
    # top_single = get_most_confused_classes(confusion_matrix_single, class_list)
    # print("==> Top 10 for single frame model: \n", top_single)

    # # 3D model
    # confusion_matrix_3d = np.load("3d_conv_confusion_matrix.npy")
    # top_3d = get_most_confused_classes(confusion_matrix_3d, class_list)
    # print("==> Top 10 for 3D model: \n", top_3d)

    combine_model_analysis(
        'single_frame_prediction_matrix.npy',
        '3d_conv_prediction_matrix.npy',
        class_list,
        test,
        num_classes
    )
    confusion_matrix_comb = np.load('combined_confusion_matrix.npy')
    top_combined = get_most_confused_classes(confusion_matrix_comb, class_list)
    print("==> Top 10 for combined model: \n", top_combined)
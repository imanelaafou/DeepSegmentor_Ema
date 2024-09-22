# Author: Yahui Liu <yahui.liu@unitn.it>

"""
Calculate sensitivity and specificity metrics:
 - Precision
 - Recall
 - F-score
"""
import matplotlib as plt
import numpy as np
import seaborn as sns
from data_io import imread
#
# def cal_prf_metrics(pred_list, gt_list, thresh_step=0.01):
#     final_accuracy_all = []
#
#     for thresh in np.arange(0.0, 0.01, thresh_step):
#         statistics = []
#
#         for pred, gt in zip(pred_list, gt_list):
#             gt_img   = (gt/255).astype('uint8')
#             pred_img = (pred/255 > thresh).astype('uint8')
#             # calculate each image
#             statistics.append(get_statistics(pred_img, gt_img))
#
#         # get tp, fp, fn
#         tp = np.sum([v[0] for v in statistics])
#         fp = np.sum([v[1] for v in statistics])
#         fn = np.sum([v[2] for v in statistics])
#         tn = np.sum([v[3] for v in statistics])
#         matrix = [[tp, fp],
#                   [fn, tn]]
#         print(np.array(matrix))
#
#
#         # calculate precision
#         p_acc = 1.0 if tp==0 and fp==0 else tp/(tp+fp)
#         # calculate recall
#         r_acc = tp/(tp+fn)
#         # calculate f-score
#         final_accuracy_all.append([thresh, p_acc, r_acc, 2*p_acc*r_acc/(p_acc+r_acc)])
#         print(f' thresh {thresh}; Precision {p_acc}; Recall {r_acc}; FScore {2*p_acc*r_acc/(p_acc+r_acc)}')
#     return final_accuracy_all
#
# def get_statistics(pred, gt):
#     """
#     return tp, fp, fn
#     """
#     tp = np.sum((pred==1)&(gt==1))
#     fp = np.sum((pred==1)&(gt==0))
#     fn = np.sum((pred==0)&(gt==1))
#     tn = np.sum((pred==0)&(gt==0))
#     return [tp, fp, fn, tn]
import matplotlib.pyplot as plt
import numpy as np

from data_io import imread

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix with adjusted text size, blue color palette, and custom class names.
    """
    plt.figure(figsize=(4, 3))  # Maintain the size
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Adjust text for better visibility
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",  # Enhancing text visibility on blue
                     fontsize=10)  # Larger font size

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()  # Adjust layout
    plt.show()



def cal_prf_metrics(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = []

    for thresh in np.arange(0.0, 0.01, thresh_step):
        statistics = []

        for pred, gt in zip(pred_list, gt_list):
            gt_img   = (gt/255).astype('uint8')
            pred_img = (pred/255 > thresh).astype('uint8')
            statistics.append(get_statistics(pred_img, gt_img))

        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])
        tn = np.sum([v[3] for v in statistics])
        matrix = [[tp, fp],
                  [fn, tn]]
        print(np.array(matrix))

        # Plot the confusion matrix
        plt.figure(figsize=(5, 4))
        plot_confusion_matrix(np.array(matrix), classes=['Crack', 'NonCrack'],
                              title='Confusion Matrix ')
        plt.show()

        p_acc = 1.0 if tp==0 and fp==0 else tp/(tp+fp)
        r_acc = tp/(tp+fn)
        final_accuracy_all.append([thresh, p_acc, r_acc, 2*p_acc*r_acc/(p_acc+r_acc)])
        print(f' thresh {thresh}; Precision {p_acc}; Recall {r_acc}; FScore {2*p_acc*r_acc/(p_acc+r_acc)}')
    return final_accuracy_all

def get_statistics(pred, gt):
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    tn = np.sum((pred==0)&(gt==0))
    return [tp, fp, fn, tn]

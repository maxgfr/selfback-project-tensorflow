print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x),
                         horizontalalignment='center',
                         verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


cm = np.array([[78, 0, 0, 0, 4, 4],
               [1, 233, 0, 1, 1, 1],
               [0, 0, 211, 13, 0, 3],
               [0, 0, 3, 171, 1, 0],
               [8, 0, 0, 2, 51, 34],
               [8, 0, 0, 2, 13, 157]])

plot_confusion_matrix(cm, ['A', 'B', 'C', 'D', 'E', 'F'])
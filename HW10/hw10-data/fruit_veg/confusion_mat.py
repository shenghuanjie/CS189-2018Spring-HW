from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import IPython

from utils import bmatrix

def main():
    """
    Result
    Plot RANDOM confusion matrix (true labels vs. predicted labels)
    """
    true_labels = [random.randint(1, 10) for i in range(100)]
    predicted_labels = [random.randint(1, 10) for i in range(100)]

    # Plot confusion matrix (true labels vs. predicted labels)
    plot = getConfusionMatrixPlot(true_labels, predicted_labels)
    plot.show()


def getConfusionMatrix(true_labels, predicted_labels):
    """
    Input
    true_labels: actual labels
    predicted_labels: model's predicted labels

    Output
    cm: confusion matrix (true labels vs. predicted labels)
    """

    # Generate confusion matrix using sklearn.metrics
    cm = confusion_matrix(true_labels, predicted_labels)
    return cm


def plotConfusionMatrix(cm, alphabet):
    """
    Input
    cm: confusion matrix (true labels vs. predicted labels)
    alphabet: names of class labels

    Output
    Plot confusion matrix (true labels vs. predicted labels)
    """

    fig = plt.figure()
    # margin for linear_classification
    #fig.subplots_adjust(bottom=0.23)
    # larger margin for hyper_search
    fig.subplots_adjust(bottom=0.23)
    plt.tight_layout()
    plt.clf()                       # Clear plot
    ax = fig.add_subplot(111)       # Add 1x1 grid, first subplot
    ax.set_aspect(1)
    res = ax.imshow(cm, cmap=plt.cm.binary,
                    interpolation='nearest', vmin=0, vmax=80)

    plt.colorbar(res)               # Add color bar

    width = len(cm)                 # Width of confusion matrix
    height = len(cm[0])             # Height of confusion matrix

    # Annotate confusion entry with numeric value
    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center', color=getFontColor(cm[x][y]))


    # Plot confusion matrix (true labels vs. predicted labels)
    plt.xticks(range(width), alphabet[:width], rotation=90)
    plt.yticks(range(height), alphabet[:height])
    #plt.show()
    return plt


def getConfusionMatrixPlot(true_labels, predicted_labels, alphabet):
    """
    Input
    true_labels: actual labels
    predicted_labels: model's predicted labels
    alphabet: names of class labels

    Output
    Plot confusion matrix (true labels vs. predicted labels)
    """

    # Generate confusion matrix using sklearn.metrics
    cm = confusion_matrix(true_labels, predicted_labels)
    print(bmatrix(cm))

    # Plot confusion matrix (true labels vs. predicted labels)
    return plotConfusionMatrix(cm, alphabet)


def getFontColor(value):
    """
    Input
    value: confusion entry value

    Output
    font color for confusion entry
    """
    if value < -1:
        return "black"
    else:
        return "white"


if __name__ == "__main__":
    main()

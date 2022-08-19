import random
import numpy as np
import matplotlib.pyplot as plt

from get_data import get_data
from models import DecisionTree, node_score_error, node_score_entropy, node_score_gini


def loss_plot(ax, title, tree, pruned_tree, train_data, test_data):
    '''
        Example plotting code. This plots four curves: the training and testing
        average loss using tree and pruned tree.
        You do not need to change this code!
        Arguments:
            - ax: A matplotlib Axes instance.
            - title: A title for the graph (string)
            - tree: An unpruned DecisionTree instance
            - pruned_tree: A pruned DecisionTree instance
            - train_data: Training dataset returned from get_data
            - test_data: Test dataset returned from get_data
    '''
    fontsize=8
    ax.plot(tree.loss_plot_vec(train_data), label='train non-pruned')
    ax.plot(tree.loss_plot_vec(test_data), label='test non-pruned')
    ax.plot(pruned_tree.loss_plot_vec(train_data), label='train pruned')
    ax.plot(pruned_tree.loss_plot_vec(test_data), label='test pruned')


    ax.locator_params(nbins=3)
    ax.set_xlabel('number of nodes', fontsize=fontsize)
    ax.set_ylabel('loss', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    legend = ax.legend(loc='upper center', shadow=True, fontsize=fontsize-2)

def explore_dataset(filename, class_name):
    train_data, validation_data, test_data = get_data(filename, class_name)

    dict_gain_functions = { "node_score_error": node_score_error,
                            "node_score_entropy": node_score_entropy,
                            "node_score_gini": node_score_gini}
    losses = []
    max_depth = list(range(1, 16))
    for i in max_depth:
        tree = DecisionTree(data=train_data, gain_function=dict_gain_functions["node_score_entropy"], max_depth = i)
        losses.append(tree.loss(train_data))
    plt.plot(max_depth, losses)
    plt.xlabel("max_depth")
    plt.ylabel("loss")
    plt.show()

    # i = 1
    # for function in list(dict_gain_functions.keys()):
    #     tree = DecisionTree(data = train_data, gain_function = dict_gain_functions[function])
    #     print("{}. Average training loss (not-pruned) with function {}: {}".format(i,function,tree.loss(train_data)))
    #     print(" {}. Average test loss (not-pruned) with function {}: {}".format( i +1,function, tree.loss(test_data)))
    #     tree_validation = DecisionTree(data = train_data, validation_data=validation_data, gain_function = dict_gain_functions[function])
    #     print(" {}. Average training loss (pruned) with function {}: {}".format(i +2,function, tree_validation.loss(train_data)))
    #     print(" {}. Average test loss (pruned) with function {}: {}".format(i + 3, function, tree_validation.loss(test_data)))
    #     i = i + 4

def main():
    ########### PLEASE DO NOT CHANGE THESE LINES OF CODE! ###################
    random.seed(1)
    np.random.seed(1)
    #########################################################################

    # explore_dataset('data/chess.csv', 'won')
    explore_dataset('data/spam.csv', '1')

# main()

if __name__ == "__main__":
    main()
"""
criterion
"""

import math


def get_criterion_function(criterion):
    if criterion == "info_gain":
        return __info_gain
    elif criterion == "info_gain_ratio":
        return __info_gain_ratio
    elif criterion == "gini":
        return __gini_index
    elif criterion == "error_rate":
        return __error_rate


def __label_stat(y, l_y, r_y):
    """Count the number of labels of nodes"""
    left_labels = {}
    right_labels = {}
    all_labels = {}
    for t in y.reshape(-1):
        if t not in all_labels:
            all_labels[t] = 0
        all_labels[t] += 1
    for t in l_y.reshape(-1):
        if t not in left_labels:
            left_labels[t] = 0
        left_labels[t] += 1
    for t in r_y.reshape(-1):
        if t not in right_labels:
            right_labels[t] = 0
        right_labels[t] += 1

    return all_labels, left_labels, right_labels


def __entropy(labels):
    """Calculate the entropy of a set of labels"""
    total_count = sum(labels.values())
    entropy = 0.0
    for count in labels.values():
        probability = count / total_count
        if probability > 0:
            entropy -= probability * math.log2(probability)
    
    return entropy


def __info_gain(y, l_y, r_y):
    """
    Calculate the info gain

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain if splitting y into      #
    # l_y and r_y                                                             #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    
    # Entropy of the parent node
    total_entropy = __entropy(all_labels)
    
    # Entropy of the left and right child nodes
    left_entropy = __entropy(left_labels)
    right_entropy = __entropy(right_labels)
    
    left_weight = len(l_y) / len(y)
    right_weight = len(r_y) / len(y)
    weighted_child_entropy = left_weight * left_entropy + right_weight * right_entropy
    
    info_gain = total_entropy - weighted_child_entropy
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return info_gain


def __info_gain_ratio(y, l_y, r_y):
    """
    Calculate the info gain ratio

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    info_gain = __info_gain(y, l_y, r_y)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain ratio if splitting y     #
    # into l_y and r_y                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    total_count = sum(all_labels.values())
    
    left_weight = len(l_y) / total_count
    right_weight = len(r_y) / total_count
    
    # Avoid devision by zero
    if left_weight == 0 or right_weight == 0:
        return 0
    
    split_info = - (left_weight * math.log2(left_weight) + right_weight * math.log2(right_weight))
    
    # Avoid devision by zero
    if split_info != 0:
        info_gain = info_gain / split_info
    else:
        info_gain = 0
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return info_gain




def __gini_index(y, l_y, r_y):
    """
    Calculate the gini index

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the gini index value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    
    def gini(labels):
        total_count = sum(labels.values())
        gini = 1.0
        for count in labels.values():
            probability = count / total_count
            gini -= probability ** 2
        return gini

    before = gini(all_labels)
    left_weight = len(l_y) / len(y)
    right_weight = len(r_y) / len(y)
    after = left_weight * gini(left_labels) + right_weight * gini(right_labels)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after


def __error_rate(y, l_y, r_y):
    """Calculate the error rate"""
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the error rate value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    def error(labels):
        total_count = sum(labels.values())
        if total_count == 0:
            return 0
        majority_class_count = max(labels.values())
        return 1 - (majority_class_count / total_count)
    
    before = error(all_labels)
    left_weight = len(l_y) / len(y)
    right_weight = len(r_y) / len(y)
    after = left_weight * error(left_labels) + right_weight * error(right_labels)
        
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after

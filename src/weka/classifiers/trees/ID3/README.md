# ID3 Classifier

ID3 (Iterative Dichotomiser 3) is a decision tree algorithm used for classification tasks. It builds a decision tree by recursively splitting the dataset based on the attribute that provides the highest information gain.

Pseudo-code for ID3 Algorithm:

``` pascal
function ID3(examples, attributes, target_attribute):
    if all examples have the same target attribute value:
        return a leaf node with that value

    if attributes is empty:
        return a leaf node with the majority target attribute value

    best_attribute = attribute with highest information gain from examples
    tree = new decision tree node with best_attribute

    for each value in best_attribute:
        subset = examples where best_attribute equals value
        if subset is empty:
            child_node = a leaf node with majority target attribute value
        else:
            child_node = ID3(subset, attributes - best_attribute, target_attribute)
        add child_node to tree under best_attribute

    return tree
```

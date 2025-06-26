package weka.classifiers.trees.models;

import weka.core.Instance;

public abstract class TreeNode {
    public abstract String evaluate(Instance instance);
}

package weka.classifiers.trees.models;

import weka.core.Instance;

public class LeafNode extends TreeNode {
    private final String value;

    public LeafNode(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }

    @Override
    public String evaluate(Instance instance) {
        return value;
    }
}

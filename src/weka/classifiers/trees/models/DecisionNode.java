package weka.classifiers.trees.models;

import weka.core.Attribute;
import weka.core.Instance;

import java.util.HashMap;
import java.util.Map;

public class DecisionNode extends TreeNode {
    private final Attribute attribute;
    private final Map<String, TreeNode> children = new HashMap<>();

    public DecisionNode(Attribute attribute) {
        this.attribute = attribute;
    }

    public void addChild(String value, TreeNode child) {
        children.put(value, child);
    }

    public Attribute getAttribute() {
        return attribute;
    }

    public Map<String, TreeNode> getChildren() {
        return children;
    }

    @Override
    public String evaluate(Instance instance) {
        TreeNode child = children.get(instance.stringValue(attribute));
        return (child != null) ? child.evaluate(instance) : "UNKNOWN";
    }
}
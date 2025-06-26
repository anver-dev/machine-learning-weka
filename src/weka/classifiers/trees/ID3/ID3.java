package weka.classifiers.trees.ID3;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.helpers.InstancesHelper;
import weka.classifiers.trees.models.DecisionNode;
import weka.classifiers.trees.models.LeafNode;
import weka.classifiers.trees.models.TreeNode;
import weka.classifiers.trees.strategies.GainStrategy;
import weka.classifiers.trees.strategies.InformationGainStrategy;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.ArrayList;
import java.util.List;

public class ID3 extends AbstractClassifier {

    private TreeNode root;
    private final GainStrategy gainStrategy = new InformationGainStrategy();

    @Override
    public void buildClassifier(Instances data) {
        System.out.println("Starting to build the decision tree...");
        Instances copyData = new Instances(data);
        copyData.setClassIndex(data.numAttributes() - 1);

        List<Attribute> attributes = InstancesHelper.extractAttributes(copyData);

        root = buildTree(copyData, attributes);

        System.out.println("Decision tree built successfully!");
        System.out.println("==========================================================");
        System.out.println("Printing the decision tree:");
        printTree(root, 0);
    }

    private TreeNode buildTree(Instances data, List<Attribute> attributes) {
        if (data.numInstances() == 0) {
            return new LeafNode("UNKNOWN");
        }

        String firstClassValue = data.firstInstance().stringValue(data.classIndex());
        if (InstancesHelper.allInstancesHaveSameClass(data, firstClassValue)) {
            return new LeafNode(firstClassValue);
        }

        if (attributes.isEmpty()) {
            return new LeafNode(InstancesHelper.getMajorityClass(data));
        }

        Attribute bestAttribute = chooseBestAttribute(data, attributes);
        DecisionNode node = new DecisionNode(bestAttribute);

        bestAttribute.enumerateValues().asIterator().forEachRemaining(val -> {
            Instances subSet = InstancesHelper.extractPartitions(data, bestAttribute).get(val);

            List<Attribute> newAttributeList = new ArrayList<>(attributes);
            newAttributeList.remove(bestAttribute);

            TreeNode child = buildTree(subSet, newAttributeList);
            node.addChild((String) val, child);
        });

        return node;
    }

    private Attribute chooseBestAttribute(Instances data, List<Attribute> attributes) {
        double bestGain = -1;
        Attribute bestAttribute = null;

        for (Attribute attribute : attributes) {
            double gain = gainStrategy.calculateGain(data, attribute);
            if (gain > bestGain) {
                bestGain = gain;
                bestAttribute = attribute;
            }
        }

        return bestAttribute;
    }

    private void printTree(TreeNode node, int depth) {
        String indent = "  ".repeat(depth);
        if (node instanceof LeafNode) {
            System.out.println(indent + "Leaf: " + ((LeafNode) node).getValue());
        } else if (node instanceof DecisionNode) {
            DecisionNode decisionNode = (DecisionNode) node;
            System.out.println(indent + "Decision: " + decisionNode.getAttribute().name());
            decisionNode.getChildren().forEach((value, child) -> {
                System.out.println(indent + "  If " + decisionNode.getAttribute().name() + " = " + value + ":");
                printTree(child, depth + 2);
            });
        }
    }

    public static void main(String[] args) throws Exception {
        ID3 id3 = new ID3();

        DataSource source = new DataSource("weather.nominal.arff");
        Instances data = source.getDataSet();

        id3.buildClassifier(data);
    }
}
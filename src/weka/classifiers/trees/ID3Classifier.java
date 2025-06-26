package weka.classifiers.trees;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * @author Ana Karina Vergara Guzmán
 */
public class ID3Classifier extends AbstractClassifier {

    private Nodo root;

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

    public void printTree(Nodo node, int level) {
        String sangria = "-".repeat(level);

        if (Objects.isNull(node.atributo)) {
            // Is a leaf node
            System.out.println(sangria + "Class: " + node.valor);
        } else {
            // Is a decision node
            for (Nodo child : node.hijos) {
                System.out.println(sangria + node.atributo.name() + " => " + child.rama);
                // Recursively print the child nodes
                printTree(child, level + 1);
            }
        }
    }


    private Nodo buildTree(Instances data, List<Attribute> attributes) {
        // There are no instances to classify
        if (data.numInstances() == 0) {
            return new Nodo("UNKNOWN");
        }

        // All instances have the same class value then return a leaf node with that class value
        // This is a base case for the recursion
        String firstClassValue = data.firstInstance().stringValue(data.classIndex());
        if (InstancesHelper.allInstancesHaveSameClass(data, firstClassValue)) {
            return new Nodo(firstClassValue);
        }

        // No attributes left to split on
        // If there are no attributes left, return a leaf node with the majority class
        // This is a base case for the recursion
        if (attributes.isEmpty()) {
            return new Nodo(InstancesHelper.getMajorityClass(data));
        }

        // Choose the best attribute to split on
        Attribute bestAttribute = chooseBestAttribute(data, attributes);
        Nodo node = new Nodo(bestAttribute);

        // Create child nodes for each value of the best attribute
        bestAttribute.enumerateValues().asIterator().forEachRemaining(val -> {
            Instances subSet = new Instances(data, 0);
            data.forEach(instance -> {
                if (instance.stringValue(bestAttribute).equals(val)) {
                    subSet.add(instance);
                }
            });

            List<Attribute> newAttributeList = new ArrayList<>(attributes);
            newAttributeList.remove(bestAttribute);

            // Recursively build the subtree for this subset
            Nodo child = buildTree(subSet, newAttributeList);
            node.AgregaHijo(child, (String) val);
        });

        return node;
    }

    private Attribute chooseBestAttribute(Instances data, List<Attribute> attributes) {
        double bestGain = -1;
        Attribute bestAttribute = null;

        for (Attribute attribute : attributes) {
            double gain = calculateGain(data, attribute);
            if (gain > bestGain) {
                bestGain = gain;
                bestAttribute = attribute;
            }
        }
        
        return bestAttribute;
    }

    /**
     * Calculates the information gain for a given attribute. The formula for calculating information gain is:
     * Gain(A) = Entropy(S) - Σ (|Si| / |S|) * Entropy(Si) ; where:
     * i is the partition of S based on attribute A,
     * |Si| is the number of instances in partition Si, |S| is the total number of instances in S,
     * Entropy(S) is the entropy of the entire dataset S,
     * and Entropy(Si) is the entropy of the partition Si.
     *
     * @param data      the dataset to evaluate
     * @param attribute the attribute for which to calculate the gain
     * @return the information gain of the attribute
     */
    private double calculateGain(Instances data, Attribute attribute) {
        System.out.println("Calculating gain for attribute: " + attribute.name());
        
        double globalEntropy = calculateEntropy(data);
        System.out.println("== Entropy: " + globalEntropy);
        Map<String, Instances> partitions = InstancesHelper.extractPartitions(data, attribute);

        double weightedEntropy = calculateEntropyOfPartitions(data, partitions);
        System.out.println("== Weighted Entropy: " + weightedEntropy);
        
        return globalEntropy - weightedEntropy;
    }

    /**
     * Calculates the weighted entropy of the partitions created by splitting the dataset on a given attribute.
     * 
     * @param data the original dataset
     * @param partitions the partitions created by the attribute
     *                  where the key is the attribute value and the value is the subset of instances
     *                  corresponding to that attribute value.
     * @return the weighted entropy of the partitions
     */
    private double calculateEntropyOfPartitions(Instances data, Map<String, Instances> partitions) {
        double weightedEntropy = 0.0;
        for (Instances attributeSubset : partitions.values()) {
            if (attributeSubset.numInstances() > 0) {
                double weight = (double) attributeSubset.numInstances() / data.numInstances();
                weightedEntropy += weight * calculateEntropy(attributeSubset);
            }
        }
        return weightedEntropy;
    }

    /**
     * Calculates the entropy of a dataset.
     * The formula for calculating entropy is:
     * Entropy(S) = - Σ (p(x) * log2(p(x))) where:
     * p(x) is the proportion of instances in class x.
     * 
     * @param data the dataset for which to calculate the entropy.
     * @return the entropy of the dataset.
     */
    private double calculateEntropy(Instances data) {
        Map<String, Integer> frequency = InstancesHelper.getFrequencyByClass(data);

        double entropy = 0.0;
        for (int freq : frequency.values()) {
            double p = (double) freq / data.numInstances();
            entropy -= p * Math.log(p) / Math.log(2);
        }
        return entropy;
    }

    @Override
    public double classifyInstance(Instance instance) {
        Nodo actual = root;
        while (actual.atributo != null) {
            String valor = instance.stringValue(actual.atributo);
            actual = actual.eligeHijo(valor);
            if (actual == null) {
                return instance.dataset().classAttribute().indexOfValue("UNKNOWN");
            }
        }
        return instance.dataset().classAttribute().indexOfValue(actual.valor);
    }

    /**
     *
     */
    @Override
    public String toString() {
        // TODO Auto-generated method stub
        return super.toString();
    }

    /**
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        ID3Classifier id3 = new ID3Classifier();

        DataSource source = new DataSource("weather.nominal.arff");
        Instances data = source.getDataSet();

        // Imprimelos
        System.out.println(data);
        System.out.println("==========================================================");

        id3.buildClassifier(data);
    }

}

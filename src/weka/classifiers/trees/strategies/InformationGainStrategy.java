package weka.classifiers.trees.strategies;

import weka.classifiers.trees.helpers.InstancesHelper;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.Map;

/**
 * InformationGainStrategy implements the GainStrategy interface to calculate
 * the information gain of a given attribute based on the entropy of the dataset.
 */
public class InformationGainStrategy implements GainStrategy {
    /**
     * Calculates the information gain for a given attribute. The formula for calculating information gain is:
     * Gain(A) = Entropy(S) - Σ (|Si| / |S|) * Entropy(Si) ; where:
     * i is the partition of S based on attribute A,
     * |Si| is the number of instances in partition Si,
     * |S| is the total number of instances in S,
     * Entropy(S) is the entropy of the entire dataset S,
     * and Entropy(Si) is the entropy of the partition Si.
     *
     * @param data      the dataset to evaluate
     * @param attribute the attribute for which to calculate the gain
     * @return the information gain of the attribute
     */
    @Override
    public double calculateGain(Instances data, Attribute attribute) {
        double globalEntropy = calculateEntropy(data);
        Map<String, Instances> partitions = InstancesHelper.extractPartitions(data, attribute);
        double weightedEntropy = calculateEntropyOfPartitions(data, partitions);
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
        for (Instances subset : partitions.values()) {
            if (subset.numInstances() > 0) {
                double weight = (double) subset.numInstances() / data.numInstances();
                weightedEntropy += weight * calculateEntropy(subset);
            }
        }
        return weightedEntropy;
    }

    /**
     * Calculates the entropy of a dataset.
     * The formula for calculating entropy is:
     * Entropy (S) = - Σ (p(x) * log2(p(x))) where:
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
}

package weka.classifiers.trees.helpers;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class InstancesHelper {
    /**
     * Extracts attributes from the given Instances object, excluding the class attribute.
     *
     * @param data the Instances object from which to extract attributes
     * @return a list of attributes excluding the class attribute
     */
    public static List<Attribute> extractAttributes(Instances data) {
        List<Attribute> attributes = new ArrayList<>();
        data.enumerateAttributes().asIterator().forEachRemaining(attr -> {
            if (isNotClassIndex(attr.index(), data.classIndex())) {
                attributes.add(attr);
            }
        });

        return attributes;
    }

    private static boolean isNotClassIndex(int currentIndex, int classIndex) {
        return currentIndex != classIndex;
    }

    /**
     * Checks if all instances in the given Instances object have the same class value.
     *
     * @param data the Instances object to check
     * @return true if all instances have the same class value, false otherwise
     */
    public static boolean allInstancesHaveSameClass(Instances data, final String firstClassValue) {
        return data.stream().allMatch(instance ->
                instance.stringValue(data.classIndex()).equals(firstClassValue)
        );
    }

    /**
     * Retrieves the majority class from the given Instances object.
     *
     * @param data the Instances object from which to retrieve the majority class
     * @return the class value that appears most frequently in the dataset
     */
    public static String getMajorityClass(Instances data) {
        Map<String, Integer> classCount = getFrequencyByClass(data);

        return classCount.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .orElseThrow()
                .getKey();
    }

    /**
     * Calculates the frequency of each class in the given Instances object.
     *
     * @param data the Instances object from which to calculate class frequencies
     * @return a map where keys are class values and values are their respective frequencies
     */
    public static Map<String, Integer> getFrequencyByClass(Instances data) {
        Map<String, Integer> frequency = new HashMap<>();
        data.forEach(instance ->
                frequency.merge(instance.stringValue(data.classIndex()), 1, Integer::sum)
        );
        return frequency;
    }

    /**
     * Extracts partitions of the given Instances object based on the specified attribute.
     *
     * @param data      the Instances object to partition
     * @param attribute the attribute based on which to partition the data
     * @return a map where keys are attribute values and values are Instances objects containing instances with that attribute value
     */
    public static Map<String, Instances> extractPartitions(Instances data, Attribute attribute) {
        Map<String, Instances> partitions = new HashMap<>();
        Enumeration<Object> valuesOfAttribute = attribute.enumerateValues();
        while (valuesOfAttribute.hasMoreElements()) {
            partitions.put((String) valuesOfAttribute.nextElement(), new Instances(data, 0));
        }

        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            String val = instance.stringValue(attribute);
            partitions.get(val).add(instance);
        }
        return partitions;
    }
}

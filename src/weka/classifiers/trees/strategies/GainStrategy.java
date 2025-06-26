package weka.classifiers.trees.strategies;

import weka.core.Attribute;
import weka.core.Instances;

public interface GainStrategy {
    double calculateGain(Instances data, Attribute attribute);
}

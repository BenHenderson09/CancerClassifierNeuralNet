package com.company;

import OML.Matrix;

public class Main {
    public static void main(String[] args) {
        // Define training examples
        Dataset data = new Dataset();
        Matrix trainingInputs;
        Matrix trainingOutputs;

        // Initialize training examples
        trainingInputs = new Matrix(data.inputData);

        trainingOutputs = new Matrix(data.outputData);

        // Initialize the network with our parameters
        Network net = new Network(trainingInputs, trainingOutputs);

        // Testing network
        net.test(new double[][]{{5,4,4,5,7,10,3,2,1}}, "benign");
        net.test(new double[][]{{8,10,10,8,7,10,9,7,1}}, "malignant");
    }
}

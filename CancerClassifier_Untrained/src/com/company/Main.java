package com.company;

import com.company.Matrix;

public class Main {

    public static void main(String[] args) {
        // Define hyper parameters
        double learningRate;
        int iterations;
        int hiddenLayerNodes;

        // Define training examples
        Dataset data = new Dataset();
        Matrix trainingInputs;
        Matrix trainingOutputs;

	    // Initialize hyper parameters (you can tune these to minimise error)
        learningRate     = 0.005;
        iterations       = 70000;
        hiddenLayerNodes = 300;

        // Initialize training examples
        trainingInputs = new Matrix(data.inputData);

        trainingOutputs = new Matrix(data.outputData);

        // Initialize the network with our parameters
        Network net = new Network(learningRate, hiddenLayerNodes, iterations, trainingInputs, trainingOutputs);

        // Training network
        net.train();
    }
}

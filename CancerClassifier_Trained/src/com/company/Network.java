package com.company;

import java.util.Arrays;

public class Network {

    // Layer Matrices
    Weights weights = new Weights();

    Matrix trainingInputs;
    Matrix trainingOutputs;

    Matrix hiddenWeights;
    Matrix outputWeights;

    public Network(Matrix inputs, Matrix outputs){

        this.trainingInputs     = inputs;
        this.trainingOutputs    = outputs;

        System.out.println("INPUTS:");
        inputs.print();
        System.out.println("OUTPUTS:");
        outputs.print();
        System.out.println("");
        InitializeWeights();
    }

    // Activation Function (Sigmoid)
    Matrix sigmoid(Matrix x){
        double[][] temp = x.convertToArr();
        for (int i = 0; i < temp.length; i++){
            for (int j = 0; j < temp[0].length; j++){
                temp[i][j] = 1/(1+Math.exp(-temp[i][j]));
            }
        }
        return new Matrix(temp);
    }


    public void InitializeWeights(){
        // Setting the weights to the values stored in the weights class. These values were acquired through using the
        // training algorithm in the other folder "CancerClassifier_Untrained". You can mess around with the
        // hyper parameters and train the network to try and get a lower error, move the weight values from the trained
        // network to the weights class and you will have your own custom weights. Here we are using weights from
        // training with a very small error.
        hiddenWeights = new Matrix(weights.hiddenWeights);
        outputWeights = new Matrix(weights.outputWeights);
    }


    Matrix feedForward(Matrix input){
        Matrix hidden = sigmoid(input.matmul(hiddenWeights));
        Matrix output = sigmoid(hidden.matmul(outputWeights));
        return output;
    }

    String[][] predict(double[][] data){
        String[][] temp = new String[data.length][data[0].length];

        for (int rows = 0; rows < data.length; rows++){
            for(int columns = 0; columns < data[0].length; columns++){
                if (data[rows][columns] < .4){
                    temp[rows][columns] = "Benign";
                }else if (data[rows][columns] > .6){
                    temp[rows][columns] = "Malignant";
                }else{
                    temp[rows][columns] = "Not Sure";
                }
            }
        }
        return temp;
    }

    public void test(double[][] testData, String label){
        double[][] netOutput = feedForward(new Matrix(testData)).convertToArr();
        String[][] prediction = predict(netOutput);

        System.out.println("-------------------------------------------------------------");
        System.out.println("Test Data: " + Arrays.deepToString(testData) + " - Should predict " + label);
        System.out.println("Prediction: " + Arrays.deepToString(prediction));
    }
}
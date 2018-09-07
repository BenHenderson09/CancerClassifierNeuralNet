package com.company;

import java.util.Arrays;

public class Network {

    // Hyper parameters
    double learningRate;
    int iterations;
    int hiddenLayerNodes;

    // Layer Matrices
    Matrix trainingInputs;
    Matrix trainingOutputs;

    Matrix hiddenWeights;
    Matrix outputWeights;

    Matrix inputLayer;
    Matrix hiddenLayer;
    Matrix outputLayer;

    // Error
    Matrix error;

    public Network(double learningRate, int hiddenLayerSize, int iterations, Matrix inputs, Matrix outputs){
        this.learningRate       = learningRate;
        this.hiddenLayerNodes   = hiddenLayerSize;
        this.iterations         = iterations;
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

    // Activation function derivative
    Matrix sigDeriv(Matrix x){
        // can be mathematically denoted as x(1-x)
        return x.multiply(x.subtract(x.multiplyConstant(2)).addConstant(1));
    }

    public void InitializeWeights(){
        // Initializing our weight matrices randomly with range of -1 to 1.
        // We set the dimensions to the previous layers nodes by the current layer's nodes.
        hiddenWeights = new Matrix(trainingInputs.convertToArr()[0].length, hiddenLayerNodes);
        hiddenWeights.setRandom(-1,1);


        outputWeights = new Matrix(hiddenLayerNodes, 1);
        outputWeights.setRandom(-1,1);
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

    double averageError(Matrix error){
        double[] temp = error.mean();
        double avrg = 0;

        for (double val : temp){
            avrg += val;
        }
        avrg /= temp.length;

        return avrg;
    }

    public void test(double[][] testData, String label){
        double[][] netOutput = feedForward(new Matrix(testData)).convertToArr();
        String[][] prediction = predict(netOutput);

        System.out.println("-------------------------------------------------------------");
        System.out.println("Test Data: " + Arrays.deepToString(testData) + " - Should predict " + label);
        System.out.println("Prediction: " + Arrays.deepToString(prediction));
    }

    public void train(){
        for (int i = 0; i < iterations; i++) {
            inputLayer = trainingInputs;
            hiddenLayer = sigmoid(inputLayer.matmul(hiddenWeights));
            outputLayer = sigmoid(hiddenLayer.matmul(outputWeights));

            // Layer derivatives
            Matrix error = outputLayer.subtract(trainingOutputs);
            Matrix outputLayerDeriv   = error.multiply(sigDeriv(outputLayer));

            Matrix hiddenLayerError   = outputLayerDeriv.matmul(outputWeights.transpose());
            Matrix hiddenLayerDeriv   = hiddenLayerError.multiply(sigDeriv(hiddenLayer));

            // Synapse (weight) derivatives
            Matrix hiddenWeightsDeriv = inputLayer.transpose().matmul(hiddenLayerDeriv);
            Matrix outputWeightsDeriv = hiddenLayer.transpose().matmul(outputLayerDeriv);


            outputWeights = outputWeights.subtract(outputWeightsDeriv.multiplyConstant(learningRate));
            hiddenWeights = hiddenWeights.subtract(hiddenWeightsDeriv.multiplyConstant(learningRate));


            // Happens regularly 10 times in our network. Note, these values are printed with scientific notation
            if ( i % (iterations / 10) == 0){
                System.out.print("Error: ");
                error.print();
            }

            // Show output
            if (i == iterations-1){

                System.out.println("Average Error: "+ averageError(error));

                System.out.print("\nOutput Layer: "); outputLayer.print();
                System.out.println("");

                test(new double[][]{{5,4,4,5,7,10,3,2,1}}, "benign");
                test(new double[][]{{8,10,10,8,7,10,9,7,1}}, "malignant");


                System.out.println("-------------------------------------------------------------");
                System.out.print("\nHidden weights: "); hiddenWeights.print();
                System.out.print("\nOutput weights: "); outputWeights.print();
            }
        }
    }
}
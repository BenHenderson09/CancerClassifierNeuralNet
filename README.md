# CancerClassifierNeuralNet
This program is a neural network algorithm, tailored to detect dangerous cancer cells based on the properties of that cell.
The algorithm determines cancer cells as benign (not dangerous) or malignant (dangerous). This gives us an insight into the
pattern of how the properties of cancerous cells can be classified as being harmful. Neural networks and machine learning in general
are making amazing advancements, they are even used in medical scenarios in the real world.


### Program details:
- The classifier is coded completely from scratch apart from the use of the OML Matrix operation library
that I created myself (from scratch). It can be seen [here](https://github.com/BenHenderson09/OpenMatrixLib).

- The program uses Java, it is a good mix of usability and speed, great for neural networks.

- The program uses a 3-layer neural network with hyper parameters defined in the Main class, feel free to toy around with these
values and try to get a lower error. 

# Usage
There are two provided folders for using the program,

Trained        | Untrained   
-------------- | -------------- 

### Trained
The "Trained" folder contains a small script with a "Weights" class. This class contains the weight values that have already been
calculated by the algorithm. Simply go to the "Main" class and run the program, it should form an output looking like this:

```Java
Training Input Data: [...]
Training Output Data: [...]
--------------------------------------------------------------------------------------------------------------------------
Test Data: [[5.0, 4.0, 4.0, 5.0, 7.0, 10.0, 3.0, 2.0, 1.0]] - Should predict benign
Prediction: [[Benign]]
--------------------------------------------------------------------------------------------------------------------------
Test Data: [[8.0, 10.0, 10.0, 8.0, 7.0, 10.0, 9.0, 7.0, 1.0]] - Should predict malignant
Prediction: [[Malignant]]
--------------------------------------------------------------------------------------------------------------------------
```

### Untrained
The "Untrained" folder contains the mechanism of the entire program. It includes a "Main" class for controlling the "Network"
class. The neural network is structured in the "Network" class, it feeds the input data through the network and then 
performs backpropagation to adjust the weight values. These weight values are the key to how neural networks function.

Hyper parameters are defined in the "Main" class. Currently, with 300 hidden nodes and 70,000 iterations it will take a few
hours to train our network. These parameters can be adjusted to your specification and are almost definitely not the best 
combination of values to find the lowest error. The "Main" class looks like this:

```Java
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
```


# Dataset
The dataset used for this project is free and available online [here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)).
Thanks very much to Dr. WIlliam H. Wolberg (physician) and University of Wisconsin Hospitals.

If you are wondering how the data is formatted nicely in the "Dataset" class of the untrained folder, I used regular expression magic.
Regular expressions or RegEx for short, are wonderful short little snippets of code used to format data. They are great for finding and
replacing characters, words etc. A great website for toying with RegEx can be found at [RegExr](https://regexr.com/).



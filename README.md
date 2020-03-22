# CancerClassifierNeuralNet
This program is a neural network algorithm, tailored to detect dangerous cancer cells based on the properties of those cells.
The algorithm determines cancer cells as benign (not dangerous) or malignant (dangerous). This gives us an insight into the
pattern of how the properties of cancerous cells can be classified as being harmful.

### Data Format
```
Attribute                     Domain
--------------------------------------------
1. Clump Thickness               1 - 10
2. Uniformity of Cell Size       1 - 10
3. Uniformity of Cell Shape      1 - 10
4. Marginal Adhesion             1 - 10
5. Single Epithelial Cell Size   1 - 10
6. Bare Nuclei                   1 - 10
7. Bland Chromatin               1 - 10
8. Normal Nucleoli               1 - 10
9. Mitoses                       1 - 10

 Output                        Domain
---------------------------------------------
Class:         (0 for benign, 1 for malignant)
```

### Program details:
- The classifier is coded from scratch apart from the use of the OML Matrix operation library
I made. It can be seen [here](https://github.com/BenHenderson09/OpenMatrixLib).

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
hours to train our network. These parameters can be adjusted to your specification.

# Dataset
The dataset used for this project is free and available online [here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)).
From University of Wisconsin Hospitals.

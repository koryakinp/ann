# Ann

Machine Learning library for .NET Core.

## Installation
```
PM> Install-Package Ann.koryakinp
```
## Usage
### Configure a Network by defining the structure and meta-parametres
```
var layerConfig = new LayerConfiguration()
  .AddInputLayer(2)
  .AddHiddenLayer(5)
  .AddHiddenLayer(5)
  .AddOutputLayer(1);
```
`AddInputLayer()`,`AddOutputLayer()` and `AddHiddenLayer()`  add layers to the network configuration with specified number of neurons.
A network must have one input, one output and any number of hidden layers.
```
var networkConfig = new NetworkConfiguration(layerConfig)
{
    Momentum = 0.9,
    LearningRate = 0.1
};
var network = new Network(networkConfig);
```
### Train Model
```
double err1 = model.TrainModel(new List<double> { 0.25, 0.50 }, new List<double> { 1 });
double err2 = model.TrainModel(new List<double> { 0.75, 0.15 }, new List<double> { 0 });
double err3 = model.TrainModel(new List<double> { 0.60, 0.40 }, new List<double> { 1 });
```
Feed network with training data. 
First argument of the `TrainModel` method accepts input values.
Second argument accepts output target values for a given training example.
The weights and biases will be adjasted using Gradient Descent with Back Propagation alghorith after each training example.
An error for that particular training data will be returned.
### Use Model
```
List<double> output = UseModel(new List<double> { 0.35, 0.45 });
```
`UseModel()` Feeds input values forward only and returns predictions of the model.

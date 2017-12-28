# Ann

Machine Learning library for .NET Core.

## Installation
```
PM> Install-Package Ann.koryakinp
```
## Basic Usage
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
First argument of the `TrainModel` method accepts input values.
Second argument accepts output target values for a given training example.
The weights and biases will be adjasted using Stochastic Gradient Descent with Back Propagation alghorith.
### Use Model
```
List<double> output = UseModel(new List<double> { 0.35, 0.45 });
```
`UseModel()` accepts input values and performs forward-only pass, returns prediction of the model.
### Save Model
After you done with trainig you can save the model in JSON file for a later use:
```
network.SaveModelToJson("network-configuration.json");
var network2 = new Network("network-configuration.json");
```
## Advanced Configuration
`AddHiddenLayer()` and `AddOutputLayer()` have usefull overloads which allow for customization of the Activation function. Out of the box following activation functions supported: Logistic Sigmoid, Hyperbolic Tangent and Rectified Linear Unit.
`AddHiddenLayer(10, ActivatorType.ReluActivator)` adds hidden layer with 10 neurons and Rectified Linear Unit activation function. If activation type is not provided the layer will use Logistic Sigmoid by default.
For further customization an implementation of the `IActivator` interface can be provided.

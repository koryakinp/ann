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
var model = new Network(networkConfig);
```
### Train Model
```
double err1 = model.TrainModel(new List<double> { 0.25, 0.50 }, new List<double> { 1 });
double err2 = model.TrainModel(new List<double> { 0.75, 0.15 }, new List<double> { 0 });
double err3 = model.TrainModel(new List<double> { 0.60, 0.40 }, new List<double> { 1 });
```
First argument of the `TrainModel` method accepts input values.
Second argument accepts output target values for a given training example.
Weights and biases will be adjasted using Stochastic Gradient Descent with Back Propagation alghorith.
### Use Model
```
List<double> output = model.UseModel(new List<double> { 0.35, 0.45 });
```
`UseModel()` accepts input values and performs forward-only pass, returns prediction of the model.
### Save Model
After you done with trainig you can save the model in JSON file for a later use:
```
model.SaveModelToJson("network-configuration.json");
var model2 = new Network("network-configuration.json");
```
## Advanced Configuration
### Customizing activation function for agiven layer
`AddHiddenLayer()` and `AddOutputLayer()` have usefull overloads which allow for customization of the Activation function. Out of the box following activation functions supported: Logistic Sigmoid, Hyperbolic Tangent and Rectified Linear Unit.
`AddHiddenLayer(10, ActivatorType.ReluActivator)` adds hidden layer with 10 neurons and Rectified Linear Unit activation function. If activation type is not provided the layer will use Logistic Sigmoid by default.
For further customization an implementation of the `IActivator` interface can be provided.
### Customizing Learning Rate, Momentum and Learning Rate Decay
If no custom configuration was provided a Network will fallback to 0.1 flat learning rate, with no momentum.
There are two learning rate decay strategy supported out of the box: Exponential decay and Step decay.
Step decay reduces the learning rate by some factor every few epochs.
Exponential decay gradually reduces the learning rate in an exponential fashion. 
More info regarding the learning rate decy can be found here: http://cs231n.github.io/neural-networks-3/#anneal

An example of custom Network Configuration:
```
NetworkConfiguration nc = new NetworkConfiguration(lc)
{
    Momentum = 0.9
    LearningRateDecayer = new StepDecayer(0.1, 0.8, 1000),
};
```
For further customization you can provide custom implementation of `ILearningRateDecayer`.
## Authors
Pavel koryakin <koryakinp@koryakinp.com>
## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/koryakinp/ann/blob/master/LICENSE) for details
## Acknowledgments
- [Leon Bottou, Stochastic Gradient Descent Tricks](https://www.microsoft.com/en-us/research/publication/stochastic-gradient-tricks/).
- [Matt Mazur, A Step by Step Backpropagation Example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/).
- [Grant Sanderson, But what *is* a Neural Network?](https://youtu.be/aircAruvnKk).

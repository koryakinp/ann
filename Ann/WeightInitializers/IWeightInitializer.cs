namespace Ann
{
    /// <summary>
    /// Used to initialize neuron connection weights prior to begining of a learning process
    /// </summary>
    public interface IWeightInitializer
    {
        /// <summary>
        /// Initialize neuron connection weights based on the number of input and output connections
        /// </summary>
        /// <param name="numberOfInputs">Number of input connections</param>
        /// <param name="numberOfOutputs">Number of output connections</param>
        /// <returns>Weight value</returns>
        double InitializeWeight(int numberOfInputs, int numberOfOutputs);
    }
}

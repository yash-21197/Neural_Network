package Neural_Network;

public abstract class ActivationFunction {
	public abstract double activation(double input);
	public abstract double diff_out_net(double neuron_output);
}
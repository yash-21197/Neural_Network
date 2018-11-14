package Neural_Network;

public class Fun_Sigmoid extends ActivationFunction{
	
	public double activation(double input){
		return ((double)1/(1+Math.exp(-1*input)));
	}
	
	public double diff_out_net(double neuron_output){
		return (neuron_output*(1-neuron_output));
	}
}

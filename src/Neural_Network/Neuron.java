package Neural_Network;

import java.util.Random;

public class Neuron {
	
	private double neuron_output;
	private double delta[][];
	private Weights neuron_weights;
	private double bias;
	private static ActivationFunction activation_function;
	
	public Neuron(){
		this.activation_function = null;
		this.neuron_weights = null;
		this.delta = null;
		this.neuron_output = 0.0;
		Random ran = new Random();
		this.bias = ran.nextDouble();
		this.bias = (ran.nextBoolean()==true) ? this.bias : -1.0*this.bias;
	}
	
	public double getNeuron_output() {
		return this.neuron_output;
	}

	public void setNeuron_output(double neuron_output) {
		this.neuron_output = neuron_output;
	}

	public double[][] getDelta() {
		return this.delta;
	}

	public void setDelta(double[][] delta) {
		this.delta = delta;
	}

	public Weights getNeuron_Weights() {
		return this.neuron_weights;
	}

	public double getBias() {
		return this.bias;
	}

	public void setBias(double bias) {
		this.bias = bias;
	}

	public static ActivationFunction getActivation_function() {
		return activation_function;
	}

	public static void setActivation_function(ActivationFunction activation_function) {
		Neuron.activation_function = activation_function;
	}
	
	public void createWeights(int x,int y){
		this.neuron_weights = new Weights(x,y);
	}
}

package Neural_Network;

import java.util.Random;

public class Weights {
	
	private double weights[][];
	
	public Weights(int x,int y){
		Random ran = new Random();
		this.weights = new double[x][y];
		
		for(int i=0;i<x;i++){
			for(int j=0;j<y;j++){
				double weight = ran.nextDouble();
				weights[i][j] = (ran.nextBoolean() == true) ? weight : -1.0*weight;
			}
		}
	}

	public double[][] getWeights() {
		return this.weights;
	}
}

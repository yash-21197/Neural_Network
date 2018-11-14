package Neural_Network;

import java.io.File;

public class Main {

	public static void main(String[] args){
		
		NeuralNetwork nn = null;
		String networkfilename = "E://network.net";
		
		if(new File(networkfilename).exists()){
			nn = NeuralNetwork.loadNetwork(networkfilename);
		}else{
			nn = new NeuralNetwork();
			nn.setInputLayerDimension(2, 1);
			nn.setHiddenLayerDimension(1, 1, 2, 1);
			nn.setOutputLayerDimension(1, 1);
			nn.setNeuronFunction("sigmoid");
			nn.initNetwork();
			
			Double in[][][] = { {{0.0},{0.5}} , {{0.6},{0.1}} , {{0.4},{0.8}} , {{0.1},{0.9}} };
			Double out[][][] = { {{1.0}} , {{0.0}} , {{1.0}} , {{1.0}} };
			nn.trainNetwork(0.5,0.001,10000,in,out);
			nn.saveNetwork(networkfilename);
		}
		Double input[][][] = { { {0.7},{0.3} } , { {0.3},{0.7} } };
		Object output[][][] = nn.testNetwork(input);
		for(int i=0;i<output.length;i++)
			for(int j=0;j<output[i].length;j++)
				for(int k=0;k<output[i][j].length;k++)
					System.out.println(output[i][j][k]);
	}
}
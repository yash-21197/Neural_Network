package Neural_Network;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Scanner;

public class NeuralNetwork {
	
	private Layer input_layer;
	private Layer hidden_layer[];
	private Layer output_layer;
	private int epoch;
	private double learning_rate;
	private double threshold_error;
	private double current_error;
	
	public NeuralNetwork() {
		this.input_layer = null;
		this.output_layer = null;
		this.hidden_layer = null;
		this.epoch = 1000;
		this.learning_rate = 0.0;
		this.threshold_error = Double.MIN_VALUE;
		this.current_error = Double.MAX_VALUE;
	}
	
	public void setInputLayerDimension(int x,int y){
		this.input_layer = new Layer(x,y);
	}
	
	public void setOutputLayerDimension(int x,int y){
		this.output_layer = new Layer(x,y);
	}
	
	public void setHiddenLayerDimension(int no_of_layers, int ... input){
		if(input.length%3==0 && no_of_layers*3==input.length){
			this.hidden_layer = new Layer[no_of_layers];
			for(int i=0;i<no_of_layers;i++)
				this.hidden_layer[input[i*3]-1] = new Layer(input[(i*3)+1],input[(i*3)+2]);
		}
	}
	
	public void setNeuronFunction(String fun){
		
		if(this.input_layer == null || this.hidden_layer == null || this.output_layer == null){
			Utility.throwError("Network structure is not completed");
		}
		
		for(int i=0;i<this.hidden_layer.length;i++)
			this.hidden_layer[i].setNeuron_function(fun);
		this.output_layer.setNeuron_function(fun);
	}
	
	private void checkStructure(){
		if(this.input_layer == null || this.hidden_layer == null || this.output_layer == null){
			Utility.throwError("Network structure is not completed");
		}
	}
	
	public void initNetwork(){
		
		this.checkStructure();
		
		for(int i=0;i<this.hidden_layer.length;i++){
			if(i==0){
				this.hidden_layer[i].joinLayers(this.input_layer.getNodes().length, this.input_layer.getNodes()[0].length);
			}else{
				this.hidden_layer[i].joinLayers(this.hidden_layer[i-1].getNodes().length, this.hidden_layer[i-1].getNodes()[0].length);
			}
		}
		this.output_layer.joinLayers(this.hidden_layer[this.hidden_layer.length-1].getNodes().length, this.hidden_layer[this.hidden_layer.length-1].getNodes()[0].length);
	}
	
	private void forward_Propagation(double output[][]){
		for(int z=0;z<this.hidden_layer.length;z++){
			Layer last = (z==0) ? this.input_layer : this.hidden_layer[z-1];
			this.hidden_layer[z].layer_Forward_Propagation(last);
		}
		this.output_layer.layer_Forward_Propagation(this.hidden_layer[this.hidden_layer.length-1]);
	}
	
	private double cal_error_cost(double output[][]){
		double T_error = 0.0;
		for(int i=0;i<this.output_layer.getNodes().length;i++){
			for(int j=0;j<this.output_layer.getNodes()[i].length;j++){
				double expected_output = output[i][j];
				double actual_output = this.output_layer.getNodes()[i][j].getNeuron_output();
				T_error += Math.pow((expected_output - actual_output), 2);
			}
		}
		T_error *= 0.5;	//(double)this.output_layer.getNodes().length*(double)this.output_layer.getNodes()[0].length
		return T_error;
	}
	
	private void back_Propagation(double output[][]){
		this.output_layer.layer_Back_Propagation(-1, this.learning_rate, this.hidden_layer[this.hidden_layer.length-1], null, output);
		for(int z=this.hidden_layer.length-1;z>=0;z--){
			Layer last = (z==0) ? this.input_layer : this.hidden_layer[z-1];
			Layer next = (z==this.hidden_layer.length-1) ? this.output_layer : this.hidden_layer[z+1];			
			this.hidden_layer[z].layer_Back_Propagation(z, this.learning_rate, last, next, output);
		}
	}
	
	public void trainNetwork(double learning_rate, double threshold_error, int epoch, Object in[][][], Object out[][][]){
		
		this.checkStructure();
		
//		long start_time = Utility.getCurrentTimeStatus();
		
		this.epoch = (epoch >= 1) ? epoch : this.epoch;
		this.learning_rate = learning_rate;
		this.threshold_error = threshold_error;
		
		double input[][][] = Utility.convertToDouble(in);
		double output[][][] = Utility.convertToDouble(out);
		
		for(int e=0;e < this.epoch && this.current_error > this.threshold_error;e++){
			for(int sample=0;sample < input.length;sample++){
				for(int i=0;i<this.input_layer.getNodes().length;i++){
					for(int j=0;j<this.input_layer.getNodes()[i].length;j++){
						this.input_layer.getNodes()[i][j].setNeuron_output(input[sample][i][j]);
					}
				}
				this.forward_Propagation(output[sample]);
				this.current_error = this.cal_error_cost(output[sample]);
				this.back_Propagation(output[sample]);
			}
		}
		
//		long end_time = Utility.getCurrentTimeStatus();
//		double total_time = Utility.getTotalTime(start_time, end_time);
//		System.out.println("Total time : " + total_time);
	}
	
	public Object[][][] testNetwork(Object in[][][]){
		
		this.checkStructure();
		
		double input[][][] = Utility.convertToDouble(in);
		Double output[][][] = new Double[input.length][this.output_layer.getNodes().length][this.output_layer.getNodes()[0].length];
		
		for(int sample=0;sample<input.length;sample++){
			for(int i=0;i<input[sample].length;i++){
				for(int j=0;j<input[sample][i].length;j++){
					this.input_layer.getNodes()[i][j].setNeuron_output(input[sample][i][j]);
				}
			}
			for(int z=0;z<this.hidden_layer.length;z++){
				Layer last = (z==0) ? this.input_layer : this.hidden_layer[z-1];
				this.hidden_layer[z].layer_Forward_Propagation(last);							
			}
			this.output_layer.layer_Forward_Propagation(this.hidden_layer[this.hidden_layer.length-1]);
			for(int i=0;i<this.output_layer.getNodes().length;i++){
				for(int j=0;j<this.output_layer.getNodes()[i].length;j++){
					output[sample][i][j] = this.output_layer.getNodes()[i][j].getNeuron_output();
				}
			}
		}
		return output;
	}
	
	public void saveNetwork(String fullpath_networkname){
		try{
			if(fullpath_networkname.endsWith(".net")){
				File net_file = new File(fullpath_networkname);
				net_file.createNewFile();
				FastInputOutput fio = new FastInputOutput(new FileInputStream(new File(fullpath_networkname)), new FileOutputStream(new File(fullpath_networkname)));
				fio.println(this.input_layer.getNodes().length + " " + this.input_layer.getNodes()[0].length);
				fio.print(this.hidden_layer.length + " ");
				for(int i=0;i<this.hidden_layer.length;i++){
					fio.print((i+1) + " " + this.hidden_layer[i].getNodes().length + " " + this.hidden_layer[i].getNodes()[0].length + " ");
				}
				fio.println("");
				fio.println(this.output_layer.getNodes().length + " " + this.output_layer.getNodes()[0].length);
				fio.println(this.output_layer.getNeuron_function());
				
				for(int i=0;i<this.hidden_layer.length;i++){
					for(int j=0;j<this.hidden_layer[i].getNodes().length;j++){
						for(int k=0;k<this.hidden_layer[i].getNodes()[i].length;k++){
							double w[][] = this.hidden_layer[i].getNodes()[j][k].getNeuron_Weights().getWeights();
							for(int ii=0;ii<w.length;ii++){
								for(int jj=0;jj<w[ii].length;jj++){
									fio.print(w[ii][jj] + " ");
								}
							}
							fio.println(this.hidden_layer[i].getNodes()[j][k].getBias());
						}
					}
				}
				for(int i=0;i<this.output_layer.getNodes().length;i++){
					for(int j=0;j<this.output_layer.getNodes()[i].length;j++){
						double w[][] = this.output_layer.getNodes()[i][j].getNeuron_Weights().getWeights();
						for(int ii=0;ii<w.length;ii++){
							for(int jj=0;jj<w[ii].length;jj++){
								fio.print(w[ii][jj] + " ");
							}
						}
						fio.println(this.output_layer.getNodes()[i][j].getBias());
					}
				}
			}else{
				throw new Exception("File extension should be '.net'.");
			}
		}catch(Exception e){
			Utility.throwError(e.toString());
		}
	}
	
	public static NeuralNetwork loadNetwork(String fullpath_networkname){
		NeuralNetwork nn = new NeuralNetwork();
		try{
			Scanner sc = new Scanner(new File(fullpath_networkname));
			
			// scan input layer
			int xi = sc.nextInt();
			int yi = sc.nextInt();
			nn.setInputLayerDimension(xi, yi);
			// scan hidden layer
			int xh = sc.nextInt();
			int arr[] = new int[3*xh];
			for(int i=0;i<xh;i++){
				arr[i*3] = sc.nextInt();
				arr[(i*3)+1] = sc.nextInt();
				arr[(i*3)+2] = sc.nextInt();
			}
			nn.setHiddenLayerDimension(xh, arr);
			// scan output layer
			int xo = sc.nextInt();
			int yo = sc.nextInt();
			nn.setOutputLayerDimension(xo, yo);
			//scan neuron function
			sc.nextLine();
			nn.setNeuronFunction(sc.nextLine().trim());
			
			nn.initNetwork();
			
			// scan weights and biases
			for(int i=0;i<xh;i++){
				for(int j=0;j<nn.hidden_layer[i].getNodes().length;j++){
					for(int k=0;k<nn.hidden_layer[i].getNodes()[j].length;k++){
						double w[][] = nn.hidden_layer[i].getNodes()[j][k].getNeuron_Weights().getWeights();
						for(int ii=0;ii<w.length;ii++){
							for(int jj=0;jj<w[ii].length;jj++){
								w[ii][jj] = sc.nextDouble();
							}
						}
						nn.hidden_layer[i].getNodes()[j][k].setBias(sc.nextDouble());
					}
				}
			}
			for(int i=0;i<nn.output_layer.getNodes().length;i++){
				for(int j=0;j<nn.output_layer.getNodes()[i].length;j++){
					double w[][] = nn.output_layer.getNodes()[i][j].getNeuron_Weights().getWeights();
					for(int ii=0;ii<w.length;ii++){
						for(int jj=0;jj<w[ii].length;jj++){
							w[ii][jj] = sc.nextDouble();
						}
					}
					nn.output_layer.getNodes()[i][j].setBias(sc.nextDouble());
				}
			}
			sc.close();
//			scan by trim
//			FastInputOutput fio = new FastInputOutput(new FileInputStream(new File(fullpath_networkname+".net")), new FileOutputStream(new File(fullpath_networkname+".net")));
//			System.out.println(fio.nextLine());
		}catch(Exception e){
			Utility.throwError(e.toString());
		}
		return nn;
	}
}
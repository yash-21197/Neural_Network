package Neural_Network;

public class Layer {

	private Neuron nodes[][];
	private String neuron_function;
	
	public Layer(int x,int y){
		this.nodes = new Neuron[x][y];
		for(int i=0;i<x;i++)
			for(int j=0;j<y;j++)
				this.nodes[i][j] = new Neuron();
		this.neuron_function = "sigmoid";
		this.nodes[0][0].setActivation_function(new Fun_Sigmoid());
	}
	
	public Neuron[][] getNodes() {
		return this.nodes;
	}

	public void setNeuron_function(String neuron_function) {
		neuron_function = neuron_function.toLowerCase().trim();
		this.neuron_function = neuron_function;
		switch(neuron_function){
		case "sigmoid" :
			nodes[0][0].setActivation_function(new Fun_Sigmoid());
			break;
		default : 
			nodes[0][0].setActivation_function(new Fun_Sigmoid());
			break;
		}
	}
	
	public String getNeuron_function(){
		return this.neuron_function;
	}

	public void joinLayers(int x, int y){
		for(int i=0;i<this.nodes.length;i++){
			for(int j=0;j<this.nodes[i].length;j++){
				this.nodes[i][j].createWeights(x, y);
			}
		}
	}
	
	public void layer_Forward_Propagation(Layer last){
		int x = this.nodes.length;
		int y = this.nodes[0].length;
		FeedForwardThread fft[][] = new FeedForwardThread[x][y];
		try{
			for(int x1=0;x1<x;x1++){
				for(int y1=0;y1<y;y1++){
					fft[x1][y1] = new FeedForwardThread(last,this.nodes[x1][y1]);
					fft[x1][y1].start();
				}
			}
			for(int x1=0;x1<x;x1++){
				for(int y1=0;y1<y;y1++){
					fft[x1][y1].join();
				}
			}
		}catch(Exception e){
			System.out.println(e.toString());
		}
	}
	
	public void layer_Back_Propagation(int index, double learning_rate, Layer last, Layer next, double output[][]){
		int x = this.nodes.length;
		int y = this.nodes[0].length;
		BackPropagationThread bpt[][] = new BackPropagationThread[x][y];
		try{
			for(int x1=0;x1<x;x1++){
				for(int y1=0;y1<y;y1++){
					double out = (index==-1) ? output[x1][y1] : 0.0;
					bpt[x1][y1] = new BackPropagationThread(index,x1,y1,out,learning_rate,last,this.nodes[x1][y1], next);
					bpt[x1][y1].start();
				}
			}
			for(int x1=0;x1<x;x1++){
				for(int y1=0;y1<y;y1++){
					bpt[x1][y1].join();
				}
			}
		}catch(Exception e){
			System.out.println(e.toString());
		}
	}
}
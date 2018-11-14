package Neural_Network;

public class BackPropagationThread extends Thread{
	private int x;
	private int y;
	private int z;
	private Layer last_layer;
	private Neuron n;
	private Layer next_layer;
	private double target_output;
	private double learning_rate;
	
	public BackPropagationThread(int z,int x,int y,double target_out, double learning_rate, Layer last, Neuron n, Layer next){
		this.x = x;
		this.y = y;
		this.z = z;
		this.last_layer = last;
		this.n = n;
		this.next_layer = next;
		this.target_output = target_out;
		this.learning_rate = learning_rate;
	}
	
	public void run(){
		try{
			if(this.z==-1){
				double w[][] = this.n.getNeuron_Weights().getWeights();
				this.n.setDelta(new double[w.length][w[0].length]);
				double e1 = -1.0 * (this.target_output - this.n.getNeuron_output());
				double e2 = this.n.getActivation_function().diff_out_net(this.n.getNeuron_output());
				double e3 = 0.0;
				double delta = 0.0;
				for(int i=0;i<w.length;i++){
					for(int j=0;j<w[i].length;j++){
						e3 = this.last_layer.getNodes()[i][j].getNeuron_output();
						delta = e1*e2*e3;
						this.n.getDelta()[i][j] = delta;
						w[i][j] -= (this.learning_rate*this.n.getDelta()[i][j]);
					}
				}
				delta = e1*e2;
				this.n.setBias(this.n.getBias() - this.learning_rate*delta);				
			}else{
				double w[][] = this.n.getNeuron_Weights().getWeights();
				this.n.setDelta(new double[w.length][w[0].length]);
				double e1 = 0.0;
				double e2 = this.n.getActivation_function().diff_out_net(this.n.getNeuron_output());
				double e3 = 0.0;
				double delta = 0.0;
				for(int ii=0;ii<this.next_layer.getNodes().length;ii++){
					for(int jj=0;jj<this.next_layer.getNodes()[ii].length;jj++){
						e1 += (this.next_layer.getNodes()[ii][jj].getDelta()[this.x][this.y]*this.next_layer.getNodes()[ii][jj].getNeuron_Weights().getWeights()[this.x][this.y]);
					}
				}
				for(int i=0;i<w.length;i++){
					for(int j=0;j<w[i].length;j++){
						e3 = this.last_layer.getNodes()[i][j].getNeuron_output();
						delta = e1*e2*e3;
						this.n.getDelta()[i][j] = delta;
						w[i][j] -= (this.learning_rate*this.n.getDelta()[i][j]);
					}
				}
				delta = e1*e2;
				this.n.setBias(this.n.getBias() - this.learning_rate*delta);
			}
		}catch(Exception e){
			System.out.println(e.toString());
		}
	}
}

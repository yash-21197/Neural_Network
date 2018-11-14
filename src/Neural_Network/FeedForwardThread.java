package Neural_Network;

public class FeedForwardThread extends Thread{
	private Layer l;
	private Neuron n;
	
	public FeedForwardThread(Layer l,Neuron n){
		this.l = l;
		this.n = n;
	}
	
	public void run(){
		try{
			double sum=this.n.getBias();
			for(int i=0;i<this.l.getNodes().length;i++){
				for(int j=0;j<this.l.getNodes()[i].length;j++){
					sum += (this.n.getNeuron_Weights().getWeights()[i][j]*this.l.getNodes()[i][j].getNeuron_output());
				}
			}
			this.n.setNeuron_output(this.n.getActivation_function().activation(sum));
		}catch(Exception e){
			System.out.println (e.toString());
		}
	}
}

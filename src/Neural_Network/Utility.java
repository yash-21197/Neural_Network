package Neural_Network;

public class Utility {
	
	public static double[][][] convertToDouble(Object arr[][][]){
		double ret[][][] = new double[arr.length][arr[0].length][arr[0][0].length];
		for(int i=0;i<arr.length;i++){
			for(int j=0;j<arr[i].length;j++){
				for(int k=0;k<arr[i][j].length;k++){
					ret[i][j][k] = Double.parseDouble(arr[i][j][k].toString());
				}
			}
		}
		return ret;
	}
	
	public static void throwError(String msg){
		try{
			throw new Exception(msg);
		}catch(Exception e){
			System.out.println(e.toString());
			System.exit(1);
		}
	}
	
	public static long getCurrentTimeStatus(){
		return System.nanoTime();
	}
	
	public static double getTotalTime(long start, long end){
		double total = (double)(end-start);
		total /= (1e9);
		return total;
	}
	
}

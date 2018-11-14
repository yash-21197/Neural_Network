package Neural_Network;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.StringTokenizer;

public class FastInputOutput{
	private BufferedReader br;
	private StringTokenizer st;
	private BufferedWriter bw;
	
	public FastInputOutput(InputStream is , OutputStream os) {
		br = new BufferedReader(new InputStreamReader(is));
		bw = new BufferedWriter(new OutputStreamWriter(os));
	}
	
	String next() {
		while (st == null || !st.hasMoreElements()) {
			try {
				st = new StringTokenizer(br.readLine());
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return st.nextToken();
	}
	
	int nextInt() {
		return Integer.parseInt(next());
	}
	
	long nextLong() {
		return Long.parseLong(next());
	}
	
	float nextFloat() {
		return Float.parseFloat(next());
	}
	
	double nextDouble() {
		return Double.parseDouble(next());
	}
	
	String nextLine(){
		String str = new String("");
		
		if(st==null || !st.hasMoreTokens()){
			try{
				str= new String(br.readLine());
			} catch (IOException e){
				e.printStackTrace();
			}
		}else{
			while(st!=null && st.hasMoreTokens()){
				str=str+" "+st.nextToken();
			}
		}
		return str;
	}
	
	void print(Object obj) throws IOException{
		bw.append(""+obj);
		bw.flush();
	}
	
	void println(Object obj) throws IOException{
		bw.append(obj + "\n");
		bw.flush();
	}
}
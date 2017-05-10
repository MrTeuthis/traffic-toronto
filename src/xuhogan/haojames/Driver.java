package xuhogan.haojames;

//import xuhogan.haojames.Vector.Orientation;
import java.util.*;

public class Driver {

	public static void main(String[] args) throws DimensionMismatchException {
		int[] layers = {2,5,5,5,4};
		NeuralNet nn = new NeuralNet(layers);
		double[][] inputs = {{2,3},{7,3},{2,3},{7,3},{2,3},{7,3}};
		double[][] outputs = {{-1,2,3,3},{0,5,6,9},{-1,2,3,9},{0,5,6,7},{-1,2,3,6},{0,5,6,5}};
		
		Matrix x = new Matrix(inputs);
		x = x.elementwiseSigmoid();
		Matrix y = new Matrix(outputs);
		y = y.elementwiseSigmoid();
		//System.out.println(nn.feedForward(x));
		for (int i = 0; i < 5000; i++) {
			nn.backpropagate(x, y, 0.05);
		}

		//nn.feedForwardActivations(x);
	}
}
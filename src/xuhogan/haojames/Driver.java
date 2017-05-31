package xuhogan.haojames;

//import xuhogan.haojames.Vector.Orientation;
import java.util.*;
import java.io.*;

public class Driver {
	
	public static long iter = 0;
	public static double cost = Double.POSITIVE_INFINITY; 

	public static void main(String[] args) throws DimensionMismatchException {		
		int[] layers = {2,3,1};
		NeuralNet nn = new NeuralNet(layers);
		
		for (iter = 0; iter < 1; iter++) {
			Matrix[] things = makeTestMatrices();
			nn.backpropagate(things[0], things[1], 0.000003);
		}
		
		ArrayList<Matrix> hypotheticallyPerfectWeights = new ArrayList<Matrix>(1);
		hypotheticallyPerfectWeights.add(new Matrix(new double[][]{{0.0, 0.75, 0.75}}));
		NeuralNet nn2 = new NeuralNet(hypotheticallyPerfectWeights);
		
		System.out.print(nn.feedForward(
				new Matrix(new double[][] {{-0.5}, {1.5}}).elementwiseSigmoid()
				).getValue(0, 0)
				);
		
		System.out.println(" vs " + new Matrix(new double[][]{{-0.75}}).elementwiseSigmoid().getValue(0, 0));
		return; 
	}

	public static String printArray(double[] vals) {
		StringBuilder ret = new StringBuilder("");
		boolean hasStarted = false;
		for (double val : vals) {
			if (hasStarted) {
				ret.append(", ");
			}
			hasStarted = true; 
			ret.append(val);
		}
		return ret.toString();
	}
	
	public static Matrix[] makeTestMatrices() {
		Random rand = new Random();
		double a = rand.nextInt(101) - 50; 
		double b = rand.nextInt(101) - 50;  
		Matrix[] ret = new Matrix[2]; 
		ret[0] = new Matrix(2, 1); 
		ret[0].setValue(0, 0, a);
		ret[0].setValue(1, 0, b);
		ret[0] = ret[0].elementwiseSigmoid().transpose(); //yeah i wrote this wrong the first time
		
		ret[1] = new Matrix(1, 1); 
		ret[1].setValue(0, 0, a + b); 
		ret[1] = ret[1].elementwiseSigmoid().transpose();
		
		return ret; 
	}

	public static boolean containsValue(Matrix mat, double d) {
		for (double[] row : mat.matrix) {
			for (double val : row) {
				if (val == d) {
					return true;
				}
			}
		}
		return false; 
	}
}
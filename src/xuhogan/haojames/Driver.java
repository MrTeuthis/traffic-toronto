package xuhogan.haojames;

//import xuhogan.haojames.Vector.Orientation;
import java.util.*;
import java.io.*;

public class Driver {
	
	public static long iter = 0;
	public static double cost = Double.POSITIVE_INFINITY; 

	public static void main(String[] args) throws DimensionMismatchException {
		int[] layers = {2,5,5,5,4};
		NeuralNet nn = new NeuralNet(layers);
		double[][] inputs = {{2,3}};
		double[][] outputs = {{0,0,0,0}};
		
		Matrix x = new Matrix(inputs);
		x = x.elementwiseSigmoid();
		Matrix y = new Matrix(outputs);
		y = y.elementwiseSigmoid();
		//System.out.println(nn.feedForward(x));
		while (cost > 1.0) {
			cost = nn.backpropagate(x, y, 0.003)[0];
		}
		System.out.println();
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

	/**
	 * Makes a test case. The input matrix, the first element in the returned array, is a horizontal
	 * vector of form [x; y] and has already been normalised. The output matrix is a horizontal vector
	 * of form [x; y; x; y]. 
	 * @return
	 */
	public static Matrix[] makeTestMatrices() {
		Random rand = new Random();
		double a = rand.nextDouble(), b = rand.nextDouble(); 
		Matrix[] ret = new Matrix[2]; 
		ret[0] = new Matrix(2, 1); 
		ret[0].setValue(0, 0, a);
		ret[0].setValue(1, 0, b);
		ret[0] = ret[0].elementwiseSigmoid().transpose(); //yeah i wrote this wrong the first time
		
		ret[1] = new Matrix(4, 1); 
		ret[1].setValue(0, 0, a);
		ret[1].setValue(1, 0, b);
		ret[1].setValue(2, 0, a);
		ret[1].setValue(3, 0, b);
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
package xuhogan.haojames;

import xuhogan.haojames.Vector.Orientation;

public class Driver {

	public static void main(String[] args) throws DimensionMismatchException {
		double[] x = {1,2,3};
		double[][] i = {{1,0,0},
						{0,1,0},
						{0,0,1}};
		Vector xVert = new Vector(x, Orientation.VERTICAL);
		Vector xHor = new Vector(x, Orientation.HORIZONTAL);
		Matrix I = new Matrix(i);
		
		System.out.println(xVert.elementwiseAdd(xHor.transpose()));
		System.out.println(xVert.elementwiseSubtract(xHor.transpose()));
		System.out.println(xVert.elementwiseOperation(xHor.transpose(), (double a, double b) -> a*b));
		System.out.println(xVert.elementwiseOperation((double a) -> a + 2));
		System.out.println(I.matrixMultiply(xVert));
		
	}
}
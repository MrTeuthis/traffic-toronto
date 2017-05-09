//package xuhogan.haojames;
//
//public class Vector extends Matrix {
//	
//	public Vector() {
//		super();
//	}
//	
//	public Vector(double[][] vector) throws DimensionMismatchException {
//		if (vector.length != 1 && vector[0].length != 1) {
//			throw new DimensionMismatchException("Input is a matrix, not a vector");
//		}
//		this.matrix = vector;
//	}
//	
//	public Vector(Matrix other) throws DimensionMismatchException {
//		if (other.getArray().length != 1 && other.getArray()[0].length != 1) {
//			throw new DimensionMismatchException("Input is a matrix, not a vector");
//		}
//		this.matrix = other.getArray();
//	}
//	
//	public Matrix toMatrix() {
//		return new Matrix(this.matrix);
//	}
//	
//	public Vector(double[] vector, Orientation orientation) {
//		if (orientation == Orientation.HORIZONTAL) {
//			this.matrix = new double[1][];
//			this.matrix[0] = vector;
//		}
//		else {
//			this.matrix = new double[vector.length][1];
//			for (int i=0; i<vector.length; i++) {
//				this.matrix[i][0] = vector[i];
//			}
//		}
//	}
//	
//	public Orientation getOrientation() {
//		if (this.matrix[0].length == 1) {
//			return Orientation.VERTICAL;
//		}
//		else {
//			return Orientation.HORIZONTAL;
//		}
//	}
//	
//	public double[] getOneDimensionalArray() {
//		if (this.getOrientation() == Orientation.HORIZONTAL) {
//			return this.matrix[0];
//		}
//		else {
//			double[] vector = new double[this.matrix.length];
//			for (int i=0; i<this.matrix.length; i++) {
//				vector[i] = this.matrix[i][0];
//			}
//			return vector;
//		}
//	}
//	
//	public Vector addBias() {
///*		double[] vector = this.getOneDimensionalArray();
//		double[] vectorWithBias = new double[vector.length + 1];
//		vectorWithBias[0] = 1;
//		for (int i=1; i<vectorWithBias.length; i++) {
//			vectorWithBias[i] = vector[i-1];
//		}
//		return new Vector(vectorWithBias, this.getOrientation());*/
//	}
//	
//	
//	public boolean isVertical() {
//		if (this.matrix.length == 1) {
//			return false;
//		}
//		else {
//			return true;
//		}
//	}
//	
//	@Override
//	public String toString() {
//		String display = this.getDimensions()[0] + "x" + this.getDimensions()[1] + " vector:\n";
//		for (int i = 0; i < this.matrix.length; i++) {
//		    for (int j = 0; j < this.matrix[i].length; j++) {
//		        display += String.format("%6.2f", this.matrix[i][j]);
//		    }
//		    display += "\n";
//		}
//		return display;
//	}
//	
//	public enum Orientation {
//		VERTICAL, HORIZONTAL
//	}
//}

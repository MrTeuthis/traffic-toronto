package xuhogan.haojames;

import java.util.function.*;

public class Matrix {
	private double[][] matrix;
	
	public Matrix()
	{
	}
	
	public Matrix(double[][] matrix) {
		this.matrix = matrix;
	}
	
	public Matrix(double[] vector, boolean isVertical) {
		if (isVertical) {
			this.matrix = new double[vector.length][1];
			for (int i=0; i<vector.length; i++) {
				this.matrix[i][0] = vector[i];
			}
		}
		else {
			this.matrix = new double[1][];
			this.matrix[0] = vector;
		}
	}
	
	public Matrix(int rows, int cols) {
		this.matrix = new double[rows][cols];
	}
	
	public void setValue(int row, int col, double value) {
		this.matrix[row][col] = value;
	}
	
	public double getValue(int row, int col) {
		return this.matrix[row][col];
	}
	
	public double[][] getArray() {
		return this.matrix;
	}
	
	public double[] getVectorArray() throws DimensionMismatchException {
		double[] vector = null;
		if (this.matrix.length == 1) {
			vector = this.matrix[0];
		}
		
		else if (this.matrix[0].length == 1) {
			vector = new double[this.matrix.length];
			for (int i=0; i<matrix.length; i++) {
				vector[i] = matrix[i][0];
			}
		}
		
		else {
			String message = this.matrix.length + "x" + this.matrix[0].length + " array is not a vector";
			throw new DimensionMismatchException(message);
		}
		
		return vector;
	}
	
	public boolean isVerticalVector() throws DimensionMismatchException {
		if (this.matrix.length == 1) {
			return false;
		}
		
		else if (this.matrix[0].length == 1) {
			return true;
		}
		
		else {
			String message = this.matrix.length + "x" + this.matrix[0].length + " array is not a vector";
			throw new DimensionMismatchException(message);
		}
	}
	
	public int[] getDimensions() {
		int[] dimensions = new int[2];
		dimensions[0] = this.matrix.length;
		dimensions[1] = this.matrix[0].length;
		return dimensions;
	}
	
	public static Matrix addBiasToVector(Matrix M) throws DimensionMismatchException{
		double[] vector = M.getVectorArray();
		double[] withBias = new double[vector.length + 1];
		withBias[0] = 1;
		for (int i=1; i<vector.length + 1; i++) {
			withBias[i] = vector[i-1];
		}
		Matrix newM = new Matrix(withBias, M.isVerticalVector());
		return newM;
	}
	
	/**
	 * Return a new matrix, the product of this and other. In other words, 
	 * (this)(other), which is different from (other)(this). 
	 * @param other another matrix that has the same height and width as the transpose
	 * 		of this matrix
	 * @return the matrix product, which is a new matrix
	 */
	public Matrix matrixMultiply(Matrix other) throws DimensionMismatchException {
		if (this.getDimensions()[1] != other.getDimensions()[0]) {
			String message = "Incompatible matrix dimensions: ";
			message += this.getDimensions()[0] + "x" + this.getDimensions()[1];
			message += " and " + other.getDimensions()[0] + "x" + other.getDimensions()[1];
			throw new DimensionMismatchException(message);
		}
		Matrix product = new Matrix(this.getDimensions()[0], other.getDimensions()[1]);
		for (int i=0; i<product.getDimensions()[0]; i++) {
			for (int j=0; j<product.getDimensions()[1]; j++) {
				for (int k=0; k< this.getDimensions()[1]; k++) {
					product.setValue(i, j, product.getValue(i, j) + this.getValue(i, k) * other.getValue(k, j));
				}
			}
		}
		return product;
	}
	
	/**
	 * Return a new matrix, the product of this matrix and the scalar num. 
	 * @param scalar a scalar
	 * @return the product, which is a new matrix
	 */
	public Matrix scalarMultiply(double scalar) {
		return elementwiseOperation(scalar, (double x, double y) -> (x * y)); 
	}
	
	public Matrix sigmoidElementwise() {
		return elementwiseOperation((double z) -> NeuralNet.sigmoid(z));
	}
	
	/**
	 * Return a new matrix, the transpose of this matrix. 
	 * @return the transpose, which is a new matrix
	 */
	public Matrix transpose() {
		int r = this.matrix.length, c = this.matrix[0].length;
		double[][] newMatrix = new double[c][r];
		for (int currentRow = 0; currentRow < r; currentRow++) {
			for (int currentCol = 0; currentCol < c; currentCol++) {
				newMatrix[currentCol][currentRow] = matrix[currentRow][currentCol];
			}
		}
		return new Matrix(newMatrix);
	}
	
	public Matrix elementwiseOperation(double other, DoubleBinaryOperator operation) {
		int r = this.matrix.length, c = this.matrix[0].length;
		double[][] newMatrix = new double[r][c];
		for (int currentRow = 0; currentRow < r; currentRow++) {
			for (int currentCol = 0; currentCol < c; currentCol++) {
				newMatrix[currentRow][currentCol] = operation.applyAsDouble(matrix[currentRow][currentCol], other);
			}
		}
		return new Matrix(newMatrix);
	}
	
	public Matrix elementwiseOperation(DoubleUnaryOperator operation) {
		int r = this.matrix.length, c = this.matrix[0].length;
		double[][] newMatrix = new double[r][c];
		for (int currentRow = 0; currentRow < r; currentRow++) {
			for (int currentCol = 0; currentCol < c; currentCol++) {
				newMatrix[currentRow][currentCol] = operation.applyAsDouble(matrix[currentRow][currentCol]);
			}
		}
		return new Matrix(newMatrix);
	}
	
	@Override
	public String toString() {
		String display = this.getDimensions()[0] + "x" + this.getDimensions()[1] + " matrix:\n";
		for (int i = 0; i < this.matrix.length; i++) {
		    for (int j = 0; j < this.matrix[i].length; j++) {
		        display += this.matrix[i][j] + " ";
		    }
		    display += "\n";
		}
		return display;
	}
}

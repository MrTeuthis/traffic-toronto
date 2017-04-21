package xuhogan.haojames;

import java.util.function.DoubleBinaryOperator;

public class Matrix {
	private double[][] matrix;
	
	private Matrix(double[][] matrix) {
		this.matrix = matrix;
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
	
	public int[] getDimensions() {
		int[] dimensions = new int[2];
		dimensions[0] = this.matrix.length;
		dimensions[1] = this.matrix[0].length;
		return dimensions;
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
			throw new DimensionMismatchException();
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
	
	@Override
	public String toString() {
		String display = "";
		for (int i = 0; i < this.matrix.length; i++) {
		    for (int j = 0; j < this.matrix[i].length; j++) {
		        display += this.matrix[i][j] + " ";
		    }
		    display += "\n";
		}
		return display;
	}
}

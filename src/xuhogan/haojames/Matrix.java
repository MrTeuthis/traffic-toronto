package xuhogan.haojames;

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
	
	/**
	 * Return a new matrix, the product of this matrix and the scalar num. 
	 * @param scalar a scalar
	 * @return the product, which is a new matrix
	 */
	public Matrix scalarMultiply(int scalar) {
		int r = this.matrix.length, c = this.matrix[0].length;
		double[][] newMatrix = new double[r][c];
		for (int currentRow = 0; currentRow < r; currentRow++) {
			for (int currentCol = 0; currentCol < c; currentCol++) {
				newMatrix[currentRow][currentCol] = matrix[currentRow][currentCol] * scalar;
			}
		}
		return new Matrix(newMatrix); 
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
	
	/**
	 * Return a new matrix, the product of this and other. In other words, 
	 * (this)(other), which is different from (other)(this). 
	 * @param other another matrix that has the same height and width as the transpose
	 * 		of this matrix
	 * @return the matrix product, which is a new matrix
	 */
	public Matrix matrixMultiply(Matrix other) {
		// TODO: exception checking
		// TODO also: the rest of the function
		// Jams, I know we did this as part of an earlier lesson on moodle, but i can't
		// find my copy of the code. if you can find yours, can you put it in here instead
		// of reinventing the wheel. 
		return null;
	}
}

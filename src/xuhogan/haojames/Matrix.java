package xuhogan.haojames;

public class Matrix {
	private double[][] matrix;
	
	public Matrix(int rows, int cols) {
		this.matrix = new double[rows][cols];
	}
	
	public void setValue(int row, int col, double value) {
		this.matrix[row][col] = value;
	}
}

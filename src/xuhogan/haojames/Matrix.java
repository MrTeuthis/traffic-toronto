package xuhogan.haojames;

public class Matrix {
	private double[][] matrix;
	
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
	
	public Matrix multiply(Matrix other) throws DimensionMismatchException {
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

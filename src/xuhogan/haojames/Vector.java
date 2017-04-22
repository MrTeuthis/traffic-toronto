package xuhogan.haojames;

public class Vector extends Matrix {
	public Vector() {
		super();
	}
	
	/**
	 * Creates a vector (i.e., that has only 1 row AND/OR only 1 column)
	 * @param vector the single-dimensional array that is the vector
	 * @param isVertical whether the vector is vertical
	 */
	public Vector(double[] vector, boolean isVertical) {
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
	
	public int getLength() {
		return Math.max(matrix.length, matrix[0].length);
	}
	
	public boolean isVerticalVector() {
		if (this.matrix.length == 1) {
			return false;
		}
		
		return true;
	}
	
	public double[] getVectorArray() {
		double[] vector = null;
		if (this.matrix.length == 1) {
			vector = this.matrix[0];
		}
		
		else {
			vector = new double[this.matrix.length];
			for (int i=0; i<matrix.length; i++) {
				vector[i] = matrix[i][0];
			}
		}
		
		return vector;
	}
	
	public static Vector addBias(Vector M) {
		double[] vector = M.getVectorArray();
		double[] withBias = new double[vector.length + 1];
		withBias[0] = 1;
		for (int i=1; i<vector.length + 1; i++) {
			withBias[i] = vector[i-1];
		}
		Vector newM = new Vector(withBias, M.isVerticalVector());
		return newM;
	}
}

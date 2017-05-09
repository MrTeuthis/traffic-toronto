package xuhogan.haojames;

import java.util.Arrays;
import java.util.function.*;
import java.text.*;

public class Matrix {
	protected double[][] matrix;
	
	public Matrix()
	{
	}
	
	/**
	 * Creates a new matrix. 
	 * @param matrix the numbers inside the matrix
	 */
	public Matrix(double[][] matrix) {
		this.matrix = matrix;
	}
	
	/**
	 * Creates a new zero matrix of the given dimensions,
	 * @param rows the number of rows
	 * @param cols the number of columns
	 */ 
	public Matrix(int rows, int cols) {
		this.matrix = new double[rows][cols];
	}
	
	/**
	 * Sets the value of a particular cell in the matrix. 
	 * @param row the row of the cell
	 * @param col the column of the cell 
	 * @param value the new value of the cell
	 */
	public void setValue(int row, int col, double value) {
		this.matrix[row][col] = value;
	}
	
	/**
	 * Gets the value of a particular cell in the matrix. 
	 * @param row the row of the cell
	 * @param col the column of the cell
	 * @return the value of the cell
	 */
	public double getValue(int row, int col) {
		return this.matrix[row][col];
	}
	
	public void fill(double value) {
		for (int i=0; i<this.matrix.length; i++) {
			Arrays.fill(this.matrix[i], value);
		}
	}
	
	/**
	 * Gets the underlying array of the matrix. 
	 * @return the underlying array of the matrix
	 */
	public double[][] getArray() {
		return this.matrix;
	}

	/**
	 * Gets the dimensions of the matrix. 
	 * @return two integers in the form of an array. The first result is the number of rows, 
	 * and the second result is the number of columns. 
	 */
	public int[] getDimensions() {
		int[] dimensions = new int[2];
		dimensions[0] = this.matrix.length;
		dimensions[1] = this.matrix[0].length;
		return dimensions;
	}
	
	/**
	 * Return a new matrix, the product of this and other. In other words, 
	 * (this)(other), which is different from (other)(this). 
	 * @param other another matrix whose height is the same as this matrix's width
	 * @return the matrix product, which is a new matrix
	 * @throws DimensionMismatchException thrown when the matrices have incompatible dimensions
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
	public Matrix elementwiseScalarMultiply(double scalar) {
		return elementwiseOperation(scalar, (double x, double y) -> (x * y)); 
	}
	
	/**
	 * Applies the sigmoid to each individual cell of the old matrix. 
	 * @return a new matrix, whose cells are the sigmoid of the corresponding cells of the old matrix
	 */
	public Matrix elementwiseSigmoid() {
		return elementwiseOperation((double z) -> NeuralNet.sigmoid(z));
	}
	
	/**
	 * Applies the derivative of the sigmoid to each individual cell of the old matrix. 
	 * @return a new matrix, whose cells are the derivative of the sigmoid of the corresponding cells of the old matrix
	 */
	public Matrix elementwiseSigmoidDerivative() {
		return elementwiseOperation((double z) -> NeuralNet.sigmoidDerivative(z));
	}
	
	/**
	 * Adds the corresponding elements of two matrices together.
	 * @param other the other matrix
	 * @return a new matrix, whose cells are the sum of the corresponding cells of the old matrices
	 * @throws DimensionMismatchException thrown when the two matrices are not the same size
	 */
	public Matrix elementwiseAdd(Matrix other) throws DimensionMismatchException {
		return elementwiseOperation(other, (double x, double y) -> (x + y));
	}
	
	/**
	 * Subtracts the corresponding elements of two matrices together.
	 * @param other the other matrix
	 * @return a new matrix, whose cells are the difference of the corresponding cells of the old matrices
	 * @throws DimensionMismatchException thrown when the two matrices are not the same size
	 */
	public Matrix elementwiseSubtract(Matrix other) throws DimensionMismatchException {
		return elementwiseOperation(other, (double x, double y) -> (x - y));
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
	 * Performs an elementwise operation. 
	 * @param other the other argument of the elementwise operation. 
	 * @param operation the operation
	 * @return a new matrix, the result of the elementwise operation. 
	 */
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
	
	/**
	 * Performs an elementwise operation. 
	 * @param other the other argument of the elementwise operation. 
	 * @param operation the operation
	 * @return a new matrix, the result of the elementwise operation. 
	 * @throws DimensionMismatchException thrown when the matrices are not the same size
	 */
	public Matrix elementwiseOperation(Matrix other, DoubleBinaryOperator operation) 
			throws DimensionMismatchException {
		if (!Arrays.equals(this.getDimensions(), other.getDimensions())) {
			String message = "Incompatible matrix dimensions: ";
			message += this.getDimensions()[0] + "x" + this.getDimensions()[1];
			message += " and " + other.getDimensions()[0] + "x" + other.getDimensions()[1];
			throw new DimensionMismatchException(message);
		}
		
		int r = this.matrix.length, c = this.matrix[0].length;
		double[][] newMatrix = new double[r][c];
		for (int currentRow = 0; currentRow < r; currentRow++) {
			for (int currentCol = 0; currentCol < c; currentCol++) {
				newMatrix[currentRow][currentCol] = operation.applyAsDouble(
						matrix[currentRow][currentCol], other.getValue(currentRow, currentCol)
						);
			}
		}
		return new Matrix(newMatrix);
	}
	
	/**
	 * Performs an unary elementwise operation.
	 * @param operation the operation
	 * @return a new matrix, the result of the elementwise operation. 
	 */
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
	
	public Matrix addBias(Direction dir) {
		double[][] newMatrix; 
		if (dir == Direction.LEFT || dir == Direction.RIGHT) {
			newMatrix = new double[this.matrix.length][this.matrix[0].length + 1];
			if (dir == Direction.LEFT) {
				//Add bias to left
				for (int row=0; row<newMatrix.length; row++) {
					newMatrix[row][0] = 1;
					for (int col=1; col<newMatrix[0].length; col++) {
						newMatrix[row][col] = this.matrix[row][col-1];
					}
				}
			}
			else {
				//Add bias to right
				for (int row=0; row<newMatrix.length; row++) {
					newMatrix[row][newMatrix[0].length-1] = 1;
					for (int col=0; col<newMatrix[0].length-1; col++) {
						newMatrix[row][col] = this.matrix[row][col];
					}
				}
			}
		}
		else {
			newMatrix = new double[this.matrix.length + 1][this.matrix[0].length];
			if (dir == Direction.UP) {
				//Add bias to top
				for (int c = 0; c < newMatrix[0].length; c++) {
					newMatrix[0][c] = 1.0;
				}
				for (int row=1; row<newMatrix.length; row++) {
					for (int col=0; col<newMatrix[0].length; col++) {
						newMatrix[row][col] = this.matrix[row-1][col];
					}
				}
			}
			else { 
				//Add bias to bottom
				for (int row=0; row<newMatrix.length-1; row++) {
					for (int col=0; col<newMatrix[0].length; col++) {
						newMatrix[row][col] = this.matrix[row][col];
					}
				}
				for (int c = 0; c < newMatrix[0].length; c++) {
					newMatrix[0][newMatrix.length-1] = 1.0;
				}
			}
		}
		
		return new Matrix(newMatrix);
	}
	
	/**
	 * Finds the sum of all of the numbers in the matrix.
	 * @return the sum of all of the numbers in the matrix
	 */
	public double sum() {
		double ret = 0.0; 
		for (double row[] : matrix) {
			for (double val : row) {
				ret += val;
			}
		}
		return ret; 
	}
	
	/**
	 * Returns a new matrix, which is this[startRows:endRows, startCols:endCols].
	 * The rows and columns designated by endRows and endCols are not included in the return value.
	 * @param startRows the first row
	 * @param endRows the end row, which supports Python-style negatives
	 * @param startCols the first column
	 * @param endCols the last column, which supports Python-style negatives
	 * @return the slice
	 */
	public Matrix slice(int startRows, int endRows, int startCols, int endCols) {
		// exception checking
		boolean erred = false;
		StringBuilder msg = new StringBuilder("");
		if (endRows < startCols) {
			msg.append("endRows > startRows");
			erred = true;
		}
		if (endCols < startCols) {
			if (erred) {
				msg.append(" and ");
			}
			msg.append("endCols < startCols");
			erred = true;
		}
		if (erred) {
			msg.append(", which is not allowed");
			throw new IndexOutOfBoundsException(msg.toString());
		}
		
		// prepare python-style negatives
		if (endRows < 0) {
			endRows += matrix.length;
		}
		if (endCols < 0) {
			endRows += matrix[0].length;
		}
		
		// the actual function body
		double[][] ret = new double[endRows - startRows][endCols - startCols];
		for (int r = startRows; r < endRows; r++) {
			for (int c = startCols; c < endCols; c++) {
				ret[r - startRows][c - startCols] = matrix[r][c];
			}
		}
		return new Matrix(ret);
	}
	
	public String dimToStr() {
		return getDimensions()[0] + "," + getDimensions()[1];
	}
	
	@Override
	public String toString() {
		String display = this.getDimensions()[0] + "x" + this.getDimensions()[1] + " matrix:\n";
		for (int i = 0; i < this.matrix.length; i++) {
		    for (int j = 0; j < this.matrix[i].length; j++) {
		        display += String.format("%6.2f", this.matrix[i][j]);
		    }
		    display += "\n";
		}
		return display;
	}
	
	public double[] getOneDimensionalArray() throws DimensionMismatchException {
		if (matrix.length == 1) {
			return matrix[0];
		}
		else if (matrix[0].length == 1) {
			double[] ret = new double[matrix.length];
			for (int i = 0; i < ret.length; i++) {
				ret[i] = matrix[0][i];
			}
			return ret;
		}
		else {
			throw new DimensionMismatchException("Cannot convert non-vector to one-dimensional array");
		}
	}
	
	@Override
	public boolean equals(Object other) {
		if (other instanceof Matrix) {
			Matrix m_other = (Matrix)other;
			return matrix == m_other.matrix;
		}
		return false;
	}
}

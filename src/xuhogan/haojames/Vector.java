package xuhogan.haojames;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

public class Vector {
	protected double[] vector; 
	
	public Vector() {
		super();
	}
	
	/**
	 * Creates a vector (i.e., that has only 1 row)
	 * @param vector the single-dimensional array that is the vector
	 */
	public Vector(double[] vector) {
		this.vector = vector;
	}
	
	public int getLength() {
		return vector.length;
	}
	
	public double[] getVectorArray() {
		return vector;
	}
	
	public static Vector addBias(Vector M) {
		double[] m_vector = M.getVectorArray();
		double[] withBias = new double[m_vector.length + 1];
		withBias[0] = 1;
		for (int i=1; i<m_vector.length + 1; i++) {
			withBias[i] = m_vector[i-1];
		}
		Vector newM = new Vector(withBias);
		return newM;
	}
	
	public Matrix toMatrix(boolean isVertical) {
		if (isVertical) {
			double[][] ret = new double[vector.length][1];
			for (int i = 0; i < vector.length; i++) {
				ret[i][0] = vector[i];
			}
			return new Matrix(ret);
		}
		else {
			return new Matrix(new double[][]{ vector });
		}
	}
	
	public Matrix toMatrix() {
		return toMatrix(false);
	}
	
	public Vector elementwiseOperation(Vector other, DoubleBinaryOperator operator) throws DimensionMismatchException {
		if (other.getLength() != this.getLength()) {
			throw new DimensionMismatchException("length mismatch in vectors " + this.getLength() + ", " + other.getLength());
		}
		
		double[] ret = new double[vector.length];
		for (int i = 0; i < vector.length; i++) {
			ret[i] = operator.applyAsDouble(vector[i], other.vector[i]);
		}
		return new Vector(ret);
	}
	
	public Vector elementwiseOperation(double other, DoubleBinaryOperator operator) {
		double[] ret = new double[vector.length];
		for (int i = 0; i < vector.length; i++) {
			ret[i] = operator.applyAsDouble(vector[i], other);
		}
		return new Vector(ret);
	}
	
	public Vector elementwiseOperation(DoubleUnaryOperator operator) {		
		double[] ret = new double[vector.length];
		for (int i = 0; i < vector.length; i++) {
			ret[i] = operator.applyAsDouble(vector[i]);
		}
		return new Vector(ret);
	}
	
	public Vector scalarMultiply(double scalar) {
		return elementwiseOperation(scalar, (double x, double y) -> (x * y)); 
	}
	
	public Vector sigmoidElementwise() {
		return elementwiseOperation((double z) -> NeuralNet.sigmoid(z));
	}
	
	public Vector sigmoidDerivativeElementwise() {
		return elementwiseOperation((double z) -> NeuralNet.sigmoidDerivative(z));
	}
	
	public Vector elementwiseAdd(Vector other) throws DimensionMismatchException {
		return elementwiseOperation(other, (double x, double y) -> (x + y));
	}
	
	public Vector elementwiseSubtract(Vector other) throws DimensionMismatchException {
		return elementwiseOperation(other, (double x, double y) -> (x - y));
	}
	
	@Override
	public String toString() {
		return toMatrix().toString();
	}
}

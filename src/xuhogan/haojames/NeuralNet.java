package xuhogan.haojames;

import java.util.ArrayList;
import java.util.Random;

public class NeuralNet {
	private int[] layers;
	private ArrayList<Matrix> weights;
	
	public NeuralNet(int[] layers) {
		this.layers = layers;
		this.weights = initWeights(layers);
	}
	
	public static ArrayList<Matrix> initWeights(int[] layers) {
		Random rand = new Random();
		ArrayList<Matrix> weights = new ArrayList<Matrix>();
		for (int i=0; i<layers.length - 1; i++) {
			weights.add(new Matrix(layers[i+1], layers[i] + 1));
			for (int j=0; j<layers[i+1]; j++) {
				for (int k=0; k<layers[i] + 1; k++) {
					weights.get(i).setValue(j, k, rand.nextDouble() * 2 - 1);
				}
			}
		}
		return weights;
	}
	
	public Matrix getMatrix(int layer) {
		return this.weights.get(layer);
	}
	
	public static double sigmoid(double z) {
		return 1/(1 + Math.exp(-z));
	}
	
	/**
	 * Feeds inputs into the neural net and returns a result vector
	 * @param inputs an array of inputs
	 * @return a matrix, which is also a vertical vector, containing the results
	 * @throws DimensionMismatchException thrown when the input array is not the right size
	 */
	public Matrix feedForward(double[] inputs) throws DimensionMismatchException {
		Matrix X = new Matrix(inputs, true);
		Matrix z = new Matrix();
		for (int i=0; i<this.layers.length-1; i++) {
			X = Matrix.addBiasToVector(X);
			z = this.getMatrix(i).matrixMultiply(X);
			X = z.sigmoidElementwise();
		}
		return X;
	}
	
	/**
	 * Feeds input into the neural net, like {@code Matrix.feedForward}, but returns the result of
	 * the cost function J(theta) instead of the result. 
	 * @param inputs an array of inputs
	 * @param answerVector the expected answer
	 * @return the cost
	 * @throws DimensionMismatchException thrown when the input array is not the right size, or 
	 * when the answer vector is not the right size 
	 */
	public double cost(double[] inputs, double[] answerVector) throws DimensionMismatchException {
		if (answerVector.length != layers[layers.length - 1]) {
			throw new DimensionMismatchException(
					"Expected vector of length " + Integer.toString(layers[layers.length - 1]) + " but got " + Integer.toString(answerVector.length)
					);
		}
		
		double[] hypothesisVector = feedForward(inputs).getVectorArray();		
		double ret = 0.0;
		
		for (int i = 0; i < answerVector.length; i++) {
			double tmp = (answerVector[i] - hypothesisVector[i]);
			ret += tmp * tmp;
		}
		
		// TODO: regularise, etc. 
		
		return ret; 
	}
}
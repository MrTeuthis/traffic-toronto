package xuhogan.haojames;

import java.util.ArrayList;
import java.util.Random;

public class NeuralNet {
	private int[] layers;
	private ArrayList<Matrix> weights;
	
	public NeuralNet()
	{
	}
	
	public NeuralNet(int[] layers) {
		this.layers = layers;
		this.weights = initWeights(layers);
	}
	
	public NeuralNet(ArrayList<Matrix> weights) {
		this.weights = weights;
		int[] layers = new int[weights.size() + 1];
		layers[1] = weights.get(0).getDimensions()[1] - 1;
		for (int i=0; i<layers.length-1; i++) {
			layers[i] = weights.get(i).getDimensions()[0];
		}
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
	
	public Matrix getWeights(int layer) {
		return this.weights.get(layer);
	}
	
	public ArrayList<Matrix> getWeights() {
		return this.weights;
	}
	
	public static double sigmoid(double z) {
		return 1/(1 + Math.exp(-z));
	}

	public static double sigmoidDerivative(double z) {
		//return Math.exp(-z)/(Math.pow((1 + Math.exp(-z)), 2));
		return z * (1-z);
	}
	
	/**
	 * Feeds inputs into the neural net and returns a result vector
	 * @param x a vertical vector of inputs
	 * @return a matrix, which is also a vertical vector, containing the results
	 * @throws DimensionMismatchException thrown when the input array is not the right size
	 */
	public Vector feedForward(Vector x) throws DimensionMismatchException {
		Vector z = new Vector();
		for (int i=0; i<this.layers.length-1; i++) {
			x = Vector.addBias(x);
			z = this.getWeights(i).matrixMultiply(x).toVector();
			x = z.sigmoidElementwise().toVector();
		}
		return x;
	}
	
	public ArrayList<Vector> feedForwardActivations(Vector x) throws DimensionMismatchException {
		ArrayList<Vector> activations = new ArrayList<Vector>();
		Vector z = new Vector();
		activations.add(x);
		for (int i=0; i<this.layers.length-1; i++) {
			x = Vector.addBias(x);
			z = this.getWeights(i).matrixMultiply(x).toVector();
			x = z.sigmoidElementwise().toVector();
			activations.add(x);
		}
		return activations;
	}	

	public Matrix backpropagate(Vector x, Vector y) throws DimensionMismatchException{
		Matrix a = feedForward(x);
		NeuralNet newNN = new NeuralNet(this.weights);
		ArrayList<Vector> activations = feedForwardActivations(x);
		ArrayList<Vector> deltas = new ArrayList<Vector>(activations.size());
		//deltas.set(activations.size()-1, activations.get(activations.size()-1).matrixSubtract(y));
		for (int i=activations.size()-2; i>=1; i--) {
			deltas.set(i, activations.get(i));
		}
		return null;
	}
	
	/**
	 * Returns the result of the cost function J(theta). 
	 * @param inputs an array of inputs
	 * @param expectedOutputs the expected answer
	 * @return the cost
	 * @throws DimensionMismatchException thrown when the input array is not the right size, or 
	 * when the answer vector is not the right size 
	 */
	public double cost(double[] outputs, double[] expectedOutputs) throws DimensionMismatchException {
		if (expectedOutputs.length != layers[layers.length - 1]) {
			throw new DimensionMismatchException(
					"Expected vector of length " + Integer.toString(layers[layers.length - 1]) + " but got " + Integer.toString(expectedOutputs.length)
					);
		}
		
		double ret = 0.0;
		
		for (int i = 0; i < expectedOutputs.length; i++) {
			double tmp = (expectedOutputs[i] - outputs[i]);
			ret += tmp * tmp;
		}
		
		// TODO: regularise, etc. 
		
		return Math.sqrt(ret); 
	}
	
	/**
	 * Returns the result of the cost function J(theta). 
	 * @param inputs an array of inputs
	 * @param expectedOutputs the expected answer
	 * @return the cost
	 * @throws DimensionMismatchException thrown when the input array is not the right size, or 
	 * when the answer vector is not the right size 
	 */
	public double cost(Vector outputs, Vector expectedOutputs) throws DimensionMismatchException {
		return cost(outputs.getVectorArray(), expectedOutputs.getVectorArray());
	}
}
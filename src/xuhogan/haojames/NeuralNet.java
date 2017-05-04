package xuhogan.haojames;

import java.util.ArrayList;
import java.util.Random;

public class NeuralNet {
	private int[] layers;
	private ArrayList<Matrix> weights;
	
	public NeuralNet()
	{
	}
	
	/**
	 * Constructs a new neural net with the given layer definition. 
	 * @param layers the layer definition for the neural net. The length of the array
	 * defines the number of layers, while each integer defines the number of nodes. 
	 */
	public NeuralNet(int[] layers) {
		this.layers = layers;
		this.weights = initWeights(layers);
	}
	
	/**
	 * Constructs a new neural net with the given weights definitions. 
	 * @param weights the list of matrices that define the connections for each layer. 
	 * The 0th matrix defines the connections between the input layer and the first hidden layer, 
	 * the 1st matrix defines the connections between the first and second hidden layers, etc. 
	 */
	public NeuralNet(ArrayList<Matrix> weights) {
		this.weights = weights;
		int[] layers = new int[weights.size() + 1];
		layers[1] = weights.get(0).getDimensions()[1] - 1;
		for (int i=0; i<layers.length-1; i++) {
			layers[i] = weights.get(i).getDimensions()[0];
		}
	}
	
	/**
	 * Randomly returns a list of matrices, which can be used to represent the weights for each layer. 
	 * @param layers the layer definition 
	 * @return the list of matrices that define the connections between the layers in the net
	 */
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
	
	/**
	 * Gets the weights for a particular layer.
	 * @param layer the layer
	 * @return the weights
	 */
	public Matrix getWeights(int layer) {
		return this.weights.get(layer);
	}
	
	/**
	 * Gets all the weights. 
	 * @return all the weights in the neural net
	 */
	public ArrayList<Matrix> getWeights() {
		return this.weights;
	}
	
	/**
	 * Computes the sigmoid function, 1/(1 + exp(-z)).
	 * @param z the argument
	 * @return the result of the sigmoid funciton
	 */
	public static double sigmoid(double z) {
		return 1/(1 + Math.exp(-z));
	}
	
	/**
	 * Computes the derivative of the sigmoid function at z, the argument. 
	 * @param z the argument
	 * @return the derivative of the sigmoid function at z
	 */
	public static double sigmoidDerivative(double z) {
		//return Math.exp(-z)/(Math.pow((1 + Math.exp(-z)), 2));
		return z * (1-z);
	}
	
	/**
	 * Feeds inputs into the neural net and returns a result vector
	 * @param x a vertical vector of inputs
	 * @return a vertical vector, containing the results
	 * @throws DimensionMismatchException thrown when the input array is not the right size
	 */
	public Vector feedForward(Vector x) throws DimensionMismatchException {
		if (x.getOrientation() == Vector.Orientation.HORIZONTAL) {
			throw new DimensionMismatchException("Input vector must be vertical");
		}
		Vector z = new Vector();
		for (int i=0; i<this.layers.length-1; i++) {
			x = x.addBias();
			z = (Vector) this.getWeights(i).matrixMultiply(x);
			x = (Vector) z.elementwiseSigmoid();
		}
		return x;
	}
	
	/**
	 * Feeds inputs into the neural net and returns activations 
	 * TODO this javadoc. idk what's happening here tbh
	 * @param x a vertical vector of inputs
	 * @return the activations
	 * @throws DimensionMismatchException thrown when the input vector is not the right size
	 */
	public ArrayList<Matrix> feedForwardActivations(Matrix X) throws DimensionMismatchException {
		ArrayList<Matrix> activations = new ArrayList<Matrix>();
		Matrix Z = new Matrix();
		activations.add(X);
		for (int i=0; i<this.layers.length-1; i++) {
			X = X.addBias();
			Z = this.getWeights(i).matrixMultiply(X);
			
			Z = Z.elementwiseSigmoid();
			activations.add(X);
		}
		return activations;
	}	

	public void backpropagate(Matrix X, Matrix Y) throws DimensionMismatchException{
		//Feedforward
		ArrayList<Matrix> activations = this.feedForwardActivations(X);
		//Set up deltas
		ArrayList<Matrix> deltas = new ArrayList<Matrix>();
		for (int i=0; i<activations.size(); i++) {
			deltas.add(null);
		}
		//Calculate delta for last layer
		deltas.set(activations.size() - 1, Y.elementwiseSubtract(activations.get(activations.size() - 1)));
		//Iterate backwards to calculate other deltas
		for (int layer=activations.size()-2; layer >= 1; layer--) {
			//Calculate z
			Matrix Z = this.weights.get(layer-1).matrixMultiply(activations.get(layer-1).addBias());
			//Calculate ThetaNoBias
			Matrix Theta = this.weights.get(layer);
			double[][] noBias = new double[Theta.getDimensions()[0]][Theta.getDimensions()[1]-1];
			for (int i=0; i<noBias.length; i++) {
				for (int j=0; j<noBias[0].length; j++) {
					noBias[i][j] = Theta.getValue(i, j+1);
				}
			}
			Matrix ThetaNoBias = new Matrix(noBias);
			//Calculate delta
			deltas.set(layer, deltas.get(layer + 1).transpose().matrixMultiply(ThetaNoBias));
			deltas.set(layer, deltas.get(layer).transpose().elementwiseOperation(Z.elementwiseSigmoidDerivative(), (a, b) -> a*b));
		}
		//Set up Deltas
		ArrayList<Matrix> Deltas = new ArrayList<Matrix>();
		for (int layer=0; layer<this.weights.size(); layer++) {
			Deltas.add(new Matrix(this.weights.get(layer).getDimensions()[0], this.weights.get(layer).getDimensions()[1]));
		}
		//Calculate Deltas
		for (int layer=0; layer<activations.size() - 1; layer++) {
			Deltas.set(layer, deltas.get(layer + 1).matrixMultiply(activations.get(layer).transpose()));
		}
		for (Matrix layer: Deltas) {
			System.out.println(layer);
		}
		//Calcuate partial derivatives of Thetas
		ArrayList<Matrix> ThetaGrads = new ArrayList<Matrix>();
		
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
		return cost(outputs.getOneDimensionalArray(), expectedOutputs.getOneDimensionalArray());
	}
}

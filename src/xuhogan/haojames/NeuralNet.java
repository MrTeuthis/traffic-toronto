package xuhogan.haojames;

import java.util.ArrayList;
import java.util.Random;
import java.util.Stack;

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
	public Matrix feedForward(Matrix x) throws DimensionMismatchException {
		if (x.getDimensions()[1] > 1) {
			throw new DimensionMismatchException("Input vector must be vertical");
		}
		Matrix z = new Matrix();
		for (int i=0; i<this.layers.length-1; i++) {
			x = x.addBias(Direction.UP);
			z = this.getWeights(i).matrixMultiply(x);
			x = z.elementwiseSigmoid();
		}
		return x;
	}
	
	/**
	 * Feeds inputs into the neural net and returns the activations
	 * of each layer. 
	 * @param X a matrix representing multiple vertical vectors of inputs
	 * @return the activations
	 * @throws DimensionMismatchException thrown when the input vector is not the right size
	 */
	public ArrayList<Matrix> feedForwardActivations(Matrix X) throws DimensionMismatchException {
		ArrayList<Matrix> activations = new ArrayList<Matrix>();
		Matrix Z = new Matrix();
		activations.add(X);
		for (int i=0; i<this.layers.length-1; i++) {
			X = X.addBias(Direction.LEFT); 
			
			Z = X.matrixMultiply(this.getWeights(i).transpose());
			
			Z = Z.elementwiseSigmoid();
			activations.add(Z);
			X = Z;
		}
		return activations;
	}	
	
	/**
	 * Does the backpropagate process once on the neural net. 
	 * Specifically, it runs a feedforward to find the activations, 
	 * and backpropagates to find the error on each node. Finally, it corrects those 
	 * errors in each node using gradient descent. 
	 * @param X the inputs
	 * @param Y the expected outputs
	 * @param lambda the regularisation factor
	 * @return the costs of each of the test cases
	 * @throws DimensionMismatchException thrown when the sizes are not correct
	 */
	public double[] backpropagate(Matrix X, Matrix Y, double lambda) throws DimensionMismatchException{
		//Feedforward
		ArrayList<Matrix> activations = this.feedForwardActivations(X);
		//Set up deltas
		ArrayList<Matrix> deltas = new ArrayList<Matrix>();
		for (int i=0; i<activations.size(); i++) {
			deltas.add(null);
		}
		//Calculate delta for last layer
		deltas.set(activations.size() - 1, activations.get(activations.size() - 1).elementwiseSubtract(Y).transpose());
		//Iterate backwards to calculate other deltas
		for (int layer=activations.size()-2; layer >= 1; layer--) {
			//Calculate z
			Matrix Z = this.weights.get(layer-1).matrixMultiply(activations.get(layer-1).addBias(Direction.LEFT).transpose());
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
			Deltas.set(layer, deltas.get(layer + 1).matrixMultiply(activations.get(layer).addBias(Direction.LEFT)));
			
		}
		//Calcuate partial derivatives of Thetas
		ArrayList<Matrix> ThetaGrads = new ArrayList<Matrix>(layers.length);
		for (int layer=0; layer<activations.size() - 1; layer++) {
			Matrix regularisationFactor = weights.get(layer).elementwiseScalarMultiply(1.0/X.getDimensions()[0] * lambda); 
			//leftmost column does not get regularisation factor
			for (int r = 0; r < regularisationFactor.getDimensions()[0]; r++) {
				regularisationFactor.setValue(r, 0, 0);
			}
			
			ThetaGrads.add(Deltas.get(layer).elementwiseScalarMultiply(1.0 / X.getDimensions()[0]).elementwiseAdd(regularisationFactor));
		}
		
		//Grand finale: update the weights
		for (int layer = 0; layer < weights.size(); layer++) {
			weights.set(layer, weights.get(layer).elementwiseAdd(ThetaGrads.get(layer).elementwiseScalarMultiply(lambda)));
		}
		
		if (Driver.iter % 1000 == 0) {
			StringBuilder sb = new StringBuilder("");
			
			sb.append("activations: ");
			sb.append(activations.get(activations.size()-1));
			
			sb.append("\nthetas: ");
			for (Matrix weight : weights) {
				sb.append(weight.toString());
			}
			sb.append("\nDeltas: ");
			for (Matrix Delta : Deltas) {
				sb.append(Delta.toString());
			}
			
			System.out.println(sb.toString());
		}
		
		return costMulti(activations.get(activations.size()-1), Y, lambda); 
	}
	
	/**
	 * Returns the result of the cost function J(theta). 
	 * @param outputs the answer given by the neural net
	 * @param expectedOutputs the expected answer
	 * @param lambda the regularisation parameter
	 * @return the cost
	 * @throws DimensionMismatchException thrown when the input array is not the right size, or 
	 * when the answer vector is not the right size 
	 */
	public double cost(double[] outputs, double[] expectedOutputs, double lambda) throws DimensionMismatchException {
		if (expectedOutputs.length != layers[layers.length - 1]) {
			throw new DimensionMismatchException(
					"Expected vector of length " + Integer.toString(layers[layers.length - 1]) + " but got " + Integer.toString(expectedOutputs.length)
					);
		}
		
		//mean squared error
		double err = 0.0;
		for (int i = 0; i < expectedOutputs.length; i++) {
			double tmp = (expectedOutputs[i] - outputs[i]);
			err += tmp * tmp;
		}
		err *= (1.0/outputs.length);
		
		//regularisation 
		double regularisation = 0.0;
		for (Matrix theta : weights) {
			regularisation += theta.elementwiseOperation((q) -> (q*q)).sum();
		}
		
		return err + regularisation; 
	}
	
	/**
	 * Returns the result of the cost function J(theta) with lambda = 0.01. 
	 * @param outputs the answer given by the neural net
	 * @param expectedOutputs the expected answer
	 * @return the cost
	 * @throws DimensionMismatchException thrown when the input array is not the right size, or 
	 * when the answer vector is not the right size 
	 */
	public double cost(double[] outputs, double[] expectedOutputs) throws DimensionMismatchException {
		return cost(outputs, expectedOutputs, 0.01);
	}
	
	/**
	 * Returns the result of the cost function J(theta). 
	 * @param outputs the answer given by the neural net
	 * @param expectedOutputs the expected answer
	 * @return the cost
	 * @throws DimensionMismatchException thrown when the input array is not the right size, or 
	 * when the answer vector is not the right size 
	 */
	public double cost(Matrix outputs, Matrix expectedOutputs) throws DimensionMismatchException {
		return cost(outputs.getOneDimensionalArray(), expectedOutputs.getOneDimensionalArray());
	}
	
	/**
	 * Returns the result of the cost function J(theta). 
	 * @param outputs the answer given by the neural net
	 * @param expectedOutputs the expected answer
	 * @param lambda the regularisation parameter
	 * @return the cost
	 * @throws DimensionMismatchException thrown when the input array is not the right size, or 
	 * when the answer vector is not the right size 
	 */
	public double cost(Matrix outputs, Matrix expectedOutputs, double lambda) throws DimensionMismatchException {
		return cost(outputs.getOneDimensionalArray(), expectedOutputs.getOneDimensionalArray(), lambda);
	}
	
	/**
	 * Returns the results of the cost function J(theta), for multiple examples. 
	 * @param outputs the answers given by the neural net
	 * @param expectedOutputs the expected answers 
	 * @param lambda the regularisation factor
	 * @return the costs
	 * @throws DimensionMismatchException thrown when the inputs are not the same sizes
	 */
	public double[] costMulti(Matrix outputs, Matrix expectedOutputs, double lambda) throws DimensionMismatchException {
		if (outputs.getDimensions()[1] != expectedOutputs.getDimensions()[1] 
				|| outputs.getDimensions()[0] != expectedOutputs.getDimensions()[0] ) {
			String message = "Matrix dimensions are different: ";
			message += outputs.getDimensions()[0] + "x" + outputs.getDimensions()[1];
			message += " and " + expectedOutputs.getDimensions()[0] + "x" + expectedOutputs.getDimensions()[1];
			throw new DimensionMismatchException(message);		
		}
		double[] ret = new double[outputs.getDimensions()[1]];
		
		for (int i = 0; i < outputs.getDimensions()[1]; i++) {
			ret[i] = cost(outputs.slice(i, i+1, 0, outputs.getDimensions()[1]), expectedOutputs.slice(i, i+1, 0, expectedOutputs.getDimensions()[1]), lambda);
		}
		return ret; 
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder("");
		for (Matrix weight : weights) {
			sb.append(weight.toString());
		}
		sb.append('\n');
		for (Matrix weight : weights) {
			sb.append(weight.toString());
		}
		return sb.toString();
	}
}

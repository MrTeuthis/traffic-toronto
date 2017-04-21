package xuhogan.haojames;

public class Driver {

	public static void main(String[] args) throws DimensionMismatchException {
		int[] layers = {3,4,2};
		NeuralNet nn = new NeuralNet(layers);
		System.out.println(nn.getWeights(0));
		System.out.println(nn.getWeights(1));
		double[] inputs = {1,2,3};
		Matrix x = new Matrix(inputs, true);
		System.out.println(nn.feedForward(x));
	}

}

package xuhogan.haojames;

public class Driver {

	public static void main(String[] args) throws DimensionMismatchException {
		int[] layers = {3,4,2};
		NeuralNet nn = new NeuralNet(layers);
		System.out.println(nn.getWeights(0));
		System.out.println(nn.getWeights(1));
		double[] inputs = {1,2,3};
		Vector x = new Vector(inputs);
		System.out.println(nn.feedForward(x));
	}

}

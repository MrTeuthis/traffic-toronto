package xuhogan.haojames;

public class Driver {

	public static void main(String[] args) throws DimensionMismatchException {
		double[][] horizontalTest = new double[][] {
			{1, 2, 3, 4, 5}
		};
		double[][] verticalTest = new double[][] {
			{1}, 
			{2}, 
			{3}, 
			{4}, 
			{5}
		};
		
		Matrix matrix_hor = new Matrix(horizontalTest);
		Matrix matrix_vert = new Matrix(verticalTest);
		
		Vector vector = new Vector(new double[] {1, 2, 3, 4, 5});
		
		System.out.println("should be true/true");
		System.out.println(Boolean.toString(
				vector.toMatrix(false).equals(matrix_hor)
				));
		System.out.println(Boolean.toString(
				vector.toMatrix(true).equals(matrix_vert)
				));
		
		
		int[] layers = {3,4,2};
		NeuralNet nn = new NeuralNet(layers);
		System.out.println(nn.getWeights(0));
		System.out.println(nn.getWeights(1));
		double[] inputs = {1,2,3};
		Vector x = new Vector(inputs);
		System.out.println(nn.feedForward(x));
	}

}

package xuhogan.haojames.test;

import java.util.function.DoubleUnaryOperator;

import xuhogan.haojames.*;

/**
 * test cases
 * @author haosy
 *
 */
public class TestCases {

	/**
	 * runs the test cases
	 * @throws Throwable
	 */
	public static void testMatrix() throws Throwable {
		// equality
		Matrix equalityA = new Matrix( new double[][] {{4, 6, 3}});
		Matrix equalityB = new Matrix( new double[][] {{4, 6, 3}});
		
		assert equalityA.equals(equalityA);
		assert equalityB.equals(equalityA);
		
		// addition
		Matrix summandA = new Matrix(
				new double[][] {
					{4.0, -5.0, 3.0},
					{8.0, 1.0, 4.0}
				}
				);
		
		Matrix summandB = new Matrix(
				new double[][] {
					{2.0, 0.0, -2.0},
					{-1.0, 2.0, 4.0}
				}
				);
		
		Matrix sum = new Matrix(
				new double[][] {
					{6.0, -5.0, 1.0},
					{7.0, 3.0, 8.0}
				}
				);
		
		assert summandA.elementwiseAdd(summandB).equals(summandB.elementwiseAdd(summandA));
		assert summandA.elementwiseAdd(summandB).equals(sum);
		
		//multiplication
		Matrix factorA = new Matrix(
				new double[][] {
					{1, 2, 3},
					{4, 5, 6}
				});
		Matrix factorB = new Matrix(
				new double[][] {
					{7, 8}, 
					{9, 10}, 
					{11, 12}
		});
		
		Matrix product = new Matrix(
				new double[][] {
					{58, 64}, 
					{139, 154}
				}
				);
		
		assert factorA.matrixMultiply(factorB).equals(product);
		
		//arbitrary elementwise functions
		Matrix arg = new Matrix(
				new double[][] {
					{-2, -4}, 
					{1, 7}
				}
				);
		
		DoubleUnaryOperator duo = new DoubleUnaryOperator() {
			@Override
			public double applyAsDouble(double operand) {
				return Math.pow(operand, operand - 5);
			}
		};
		
		Matrix result = new Matrix(
				new double[][] {
					{duo.applyAsDouble(-2), duo.applyAsDouble(-4)}, 
					{duo.applyAsDouble(1), duo.applyAsDouble(7)}
				}
				);
		
		assert arg.elementwiseOperation(duo).equals(result);
	}
	
	/**
	 * runs the test cases
	 * @param args arguments
	 * @throws Throwable throwable
	 */
	public static void main(String[] args) throws Throwable {
		testMatrix();
	}
}

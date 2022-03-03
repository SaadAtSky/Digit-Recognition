package MLP;

/*
 * This Matrix class is used to represent mathematic objects called matrices 
 * and performs the mathematical operations associated with matrices
 * that are relevant to our machine learning classifiers such as MLP and SVM
 */
public class Matrix {
	private int row;
	private int column;
	private double[][] data;

	// defines the matrix's dimensions
	public Matrix(int row, int column) {
		setRow(row);
		setColumn(column);
		setData(new double[getRow()][getColumn()]);
		for (int x = 0; x < this.getRow(); x++) {
			for (int y = 0; y < this.getColumn(); y++) {
				this.getData()[x][y] = 0;
			}
		}
	}

	// Matrix dot Product
	public static Matrix dotProduct(Matrix firstMatrix, Matrix secondMatrix) {
		Matrix result = new Matrix(firstMatrix.getRow(), secondMatrix.getColumn());
		double sum = 0;
		for (int row = 0; row < result.getRow(); row++) {
			sum = 0;
			for (int column = 0; column < result.getColumn(); column++) {
				for (int index = 0; index < firstMatrix.getColumn(); index++) {
					sum = sum + firstMatrix.getData()[row][index] * secondMatrix.getData()[index][column];
				}
				result.getData()[row][column] = sum;
			}
		}
		return result;
	}

	// add matrix new column to the end of the matrix
	// containing the value '1' for each row
	public static Matrix addOnes(Matrix matrix) {
		Matrix result = new Matrix(matrix.getRow(), matrix.getColumn() + 1);
		for (int row = 0; row < result.getRow(); row++) {
			for (int column = 0; column < result.getColumn(); column++) {
				if (column == matrix.getColumn()) {
					result.getData()[row][column] = 1;
				} else {
					result.getData()[row][column] = matrix.getData()[row][column];
				}

			}
		}
		return result;
	}

	// find the absolute value using the Pythagoras Theorum
	public static double absoluteValue(Matrix matrix) {
		double sumOfSquares = 0;
		for (int row = 0; row < matrix.getRow(); row++) {
			for (int column = 0; column < matrix.getColumn(); column++) {
				sumOfSquares = sumOfSquares + Math.pow(matrix.getData()[row][column], 2);
			}
		}
		return Math.sqrt(sumOfSquares);
	}

	// multipy matrix Matrix by matrix scalar value
	public static Matrix scalarMultiply(Matrix matrix, double scalarMultiplier) {
		Matrix result = new Matrix(matrix.getRow(), matrix.getColumn());
		for (int row = 0; row < result.getRow(); row++) {
			for (int column = 0; column < result.getColumn(); column++) {
				result.getData()[row][column] = matrix.getData()[row][column] * scalarMultiplier;
			}
		}
		return result;
	}

	// element wise multiplication of two matrices
	public static Matrix elementWise(Matrix firstMatrix, Matrix secondMatrix) {
		Matrix result = new Matrix(firstMatrix.getRow(), firstMatrix.getColumn());
		for (int row = 0; row < result.getRow(); row++) {
			for (int column = 0; column < result.getColumn(); column++) {
				result.getData()[row][column] = firstMatrix.getData()[row][column] * secondMatrix.getData()[row][column];
			}
		}
		return result;
	}

	// add two matrices
	public static Matrix add(Matrix firstMatrix, Matrix secondMatrix) {
		Matrix result = new Matrix(firstMatrix.getRow(), firstMatrix.getColumn());
		for (int row = 0; row < result.getRow(); row++) {
			for (int column = 0; column < result.getColumn(); column++) {
				result.getData()[row][column] = firstMatrix.getData()[row][column] + secondMatrix.getData()[row][column];
			}
		}
		return result;
	}

	// subtract two matrices
	public static Matrix subtract(Matrix firstMatrix, Matrix secondMatrix) {
		Matrix result = new Matrix(firstMatrix.getRow(), firstMatrix.getColumn());
		for (int row = 0; row < firstMatrix.getRow(); row++) {
			result.getData()[row][0] = firstMatrix.getData()[row][0] - secondMatrix.getData()[row][0];
		}
		return result;
	}

	// subtract matrix scalar value from matrix matrix
	public static Matrix subtractScalar(Matrix matrix, double scalarValue) {
		Matrix result = new Matrix(matrix.getRow(), matrix.getColumn());
		for (int row = 0; row < matrix.getRow(); row++) {
			result.getData()[row][0] = matrix.getData()[row][0] - scalarValue;
		}
		return result;
	}

	// flip the matrix so that rows = columns and column = rows
	public static Matrix transpose(Matrix matrix) {
		Matrix result = new Matrix(matrix.getColumn(), matrix.getRow());
		for (int row = 0; row < result.getRow(); row++) {
			for (int column = 0; column < result.getColumn(); column++) {
				result.getData()[row][column] = matrix.getData()[column][row];
			}
		}
		return result;
	}

	// convert two-dimensional array to matrix
	public static Matrix fromArray(double[][] array) {
		Matrix result = new Matrix(array.length, array[0].length);
		for (int row = 0; row < result.getRow(); row++) {
			for (int column = 0; column < result.getColumn(); column++) {
				result.getData()[row][column] = array[row][column];
			}
		}
		return result;
	}

	// convert one-dimensional array with integral values
	// to matrix with column value as 1
	public static Matrix from1DArray(int[] array) {
		Matrix result = new Matrix(array.length, 1);
		for (int row = 0; row < result.getRow(); row++) {
			result.getData()[row][0] = array[row];
		}
		return result;
	}

	// convert one-dimensional array with double value
	// to matrix with column value as 1
	public static Matrix from1DArrayDouble(double[] array) {
		Matrix result = new Matrix(array.length, 1);
		for (int row = 0; row < result.getRow(); row++) {
			result.getData()[row][0] = array[row];
		}
		return result;
	}

	// convert matrix to array with the same dimensions
	public static double[][] toArray(Matrix matrix) {
		double[][] result = new double[matrix.getRow()][matrix.getColumn()];
		for (int row = 0; row < matrix.getRow(); row++) {
			for (int column = 0; column < matrix.getColumn(); column++) {
				result[row][column] = matrix.getData()[row][column];
			}
		}
		return result;
	}

	// display matrix data
	public void print() {
		System.out.println("------------");
		for (int row = 0; row < this.getRow(); row++) {
			for (int column = 0; column < this.getColumn(); column++) {
				System.out.print(this.getData()[row][column] + " ");
			}
			System.out.println();
		}
		System.out.println("------------");
	}

	public int getRow() {
		return row;
	}

	public void setRow(int row) {
		this.row = row;
	}

	public int getColumn() {
		return column;
	}

	public void setColumn(int column) {
		this.column = column;
	}

	public double[][] getData() {
		return data;
	}

	public void setData(double[][] data) {
		this.data = data;
	}
}

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

	// defines the matix's dimensions
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
	public static Matrix dotProduct(Matrix m1, Matrix m2) {
		Matrix result = new Matrix(m1.getRow(), m2.getColumn());
		double sum = 0;
		for (int row = 0; row < result.getRow(); row++) {
			sum = 0;
			for (int column = 0; column < result.getColumn(); column++) {
				for (int index = 0; index < m1.getColumn(); index++) {
					sum = sum + m1.getData()[row][index] * m2.getData()[index][column];
				}
				result.getData()[row][column] = sum;
			}
		}
		return result;
	}

	// add m1 new column to the end of the matrix
	// containing the value '1' for each row
	public static Matrix addOnes(Matrix m1) {
		Matrix result = new Matrix(m1.getRow(), m1.getColumn() + 1);
		for (int row = 0; row < result.getRow(); row++) {
			for (int column = 0; column < result.getColumn(); column++) {
				if (column == m1.getColumn()) {
					result.getData()[row][column] = 1;
				} else {
					result.getData()[row][column] = m1.getData()[row][column];
				}

			}
		}
		return result;
	}

	// find the absolute value using the Pythagoras Theorum
	public static double absoluteValue(Matrix m1) {
		double sumOfSquares = 0;
		for (int row = 0; row < m1.getRow(); row++) {
			for (int column = 0; column < m1.getColumn(); column++) {
				sumOfSquares = sumOfSquares + Math.pow(m1.getData()[row][column], 2);
			}
		}
		return Math.sqrt(sumOfSquares);
	}

	// multipy m1 Matrix by m1 scalar value
	public static Matrix scalarMultiply(Matrix m1, double scalarMultiplier) {
		Matrix result = new Matrix(m1.getRow(), m1.getColumn());
		for (int row = 0; row < result.getRow(); row++) {
			for (int column = 0; column < result.getColumn(); column++) {
				result.getData()[row][column] = m1.getData()[row][column] * scalarMultiplier;
			}
		}
		return result;
	}

	// element wise multiplication of two matrices
	public static Matrix elementWise(Matrix m1, Matrix m2) {
		Matrix result = new Matrix(m1.getRow(), m1.getColumn());
		for (int row = 0; row < result.getRow(); row++) {
			for (int column = 0; column < result.getColumn(); column++) {
				result.getData()[row][column] = m1.getData()[row][column] * m2.getData()[row][column];
			}
		}
		return result;
	}

	// add two matrices
	public static Matrix add(Matrix m1, Matrix m2) {
		Matrix result = new Matrix(m1.getRow(), m1.getColumn());
		for (int row = 0; row < result.getRow(); row++) {
			for (int column = 0; column < result.getColumn(); column++) {
				result.getData()[row][column] = m1.getData()[row][column] + m2.getData()[row][column];
			}
		}
		return result;
	}

	// subtract two matrices
	public static Matrix subtract(Matrix m1, Matrix m2) {
		Matrix result = new Matrix(m1.getRow(), m1.getColumn());
		for (int row = 0; row < m1.getRow(); row++) {
			result.getData()[row][0] = m1.getData()[row][0] - m2.getData()[row][0];
		}
		return result;
	}

	// subtract m1 scalar value from m1 matrix
	public static Matrix subtractScalar(Matrix m1, double scalarValue) {
		Matrix result = new Matrix(m1.getRow(), m1.getColumn());
		for (int row = 0; row < m1.getRow(); row++) {
			result.getData()[row][0] = m1.getData()[row][0] - scalarValue;
		}
		return result;
	}

	// flip the matrix so that rows = columns and column = rows
	public static Matrix transpose(Matrix m1) {
		Matrix result = new Matrix(m1.getColumn(), m1.getRow());
		for (int row = 0; row < result.getRow(); row++) {
			for (int column = 0; column < result.getColumn(); column++) {
				result.getData()[row][column] = m1.getData()[column][row];
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
	public static double[][] toArray(Matrix m1) {
		double[][] result = new double[m1.getRow()][m1.getColumn()];
		for (int row = 0; row < m1.getRow(); row++) {
			for (int column = 0; column < m1.getColumn(); column++) {
				result[row][column] = m1.getData()[row][column];
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

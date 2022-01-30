package MLP;

public class Matrix{
	int row,column;
	double[][] data;;
	public Matrix(int r,int c) {
		row = r;
		column = c;
		data = new double[row][column];
		for(int x=0;x<this.row;x++) {
			for(int y=0;y<this.column;y++) {
				this.data[x][y] = 0;
			}
		}
	}
	
	//Matrix Product
	public static Matrix multiply(Matrix a, Matrix b) {
		Matrix result = new Matrix(a.row,b.column);
		double sum=0;
		for(int r=0;r<result.row;r++) {
			sum=0;
			for(int c=0;c<result.column;c++) {
				for(int index=0;index<a.column;index++) {
					sum = sum + a.data[r][index]*b.data[index][c];
				}
				result.data[r][c]=sum;
			}
		}
		return result;
//		System.out.println("------------");
//		for(int r=0;r<result.row;r++) {
//			for(int c=0;c<result.column;c++) {
//				System.out.print(result.data[r][c]+" ");
//			}
//			System.out.println();
//		}
//		System.out.print("------------");
	}
	
	//scalar multiplication
	public static Matrix scalarMultiply(Matrix a,double scalarMultiplier) {
		Matrix result = new Matrix(a.row,a.column);
		for(int row = 0;row<result.row;row++) {
			for(int column = 0;column <result.column;column++) {
				result.data[row][column] = a.data[row][column]*scalarMultiplier;
			}
		}
		return result;
	}
	
	//add two matrix
	public static Matrix elementWise(Matrix a,Matrix b) {
		Matrix result = new Matrix(a.row,a.column);
		for(int row = 0;row<result.row;row++) {
			for(int column = 0;column <result.column;column++) {
				result.data[row][column] = a.data[row][column]*b.data[row][column];
			}
		}
		return result;
	}
	
//	public static double costFunction(Matrix target, Matrix output) {
//		double result = 0;
//		for(int row = 0;row<target.row;row++) {
//			result = result + ((Math.pow(target.data[row][0]-output.data[row][0],2))*0.5);
//		}
//		
//		return result;
//	}
	public static double costFunction(Matrix outputError) {
		double result = 0;
		for(int row = 0;row<outputError.row;row++) {
			result = result + Math.pow(outputError.data[row][0],2);
		}
		
		return result;
	}
	
	//add two matrix
	public static Matrix add(Matrix a,Matrix b) {
		Matrix result = new Matrix(a.row,a.column);
		for(int row = 0;row<result.row;row++) {
			for(int column = 0;column <result.column;column++) {
				result.data[row][column] = a.data[row][column]+b.data[row][column];
			}
		}
		return result;
	}
	
	//add bias
//	public static Matrix addScalar(Matrix a, double scalar) {
//		Matrix result = new Matrix(a.row,a.column);
//		for(int row=0;row<a.row;row++) {
//			result.data[row][0] = a.data[row][0]+scalar;
//		}
//		return result;
//	}
	
	//add bias
	public static Matrix subtract(Matrix a, Matrix b) {
		Matrix result = new Matrix(a.row,a.column);
		for(int row=0;row<a.row;row++) {
			result.data[row][0] = a.data[row][0]-b.data[row][0];
		}
		return result;
	}
	
	//flip the matrix so that rows = columns and column = rows
	public static Matrix transpose(Matrix input) {
		Matrix result = new Matrix(input.column,input.row);
		for(int row = 0;row<result.row;row++) {
			for(int column = 0;column<result.column;column++) {
				result.data[row][column] = input.data[column][row];
			}
		}
		return result;
	}
	
	public static double sigmoid(double value) {
		return 1/(1+(Math.exp(-value)));
	}
	
	public static Matrix deactivate(Matrix a) {
		Matrix result = new Matrix(a.row,a.column);
		for(int row=0;row<a.row;row++) {
			result.data[row][0] = a.data[row][0]*(1-a.data[row][0]);
		}
		return result;
	}
	
	//apply sigmoid function
	public static Matrix activation(Matrix a) {
		Matrix result = new Matrix(a.row,a.column);
		for(int row=0;row<a.row;row++) {
			result.data[row][0] = sigmoid(a.data[row][0]);
		}
		return result;
	}
	
	//array to matrix
	public static Matrix fromArray(double[][] array) {
		Matrix result = new Matrix(array.length,array[0].length);
			for(int row = 0;row<result.row;row++) {
				for(int column=0;column<result.column;column++) {
					result.data[row][column]=array[row][column];
				}
			}
		return result;
	}
	
	//matrix to array
	public static double[][] toArray(Matrix m) {
		double[][] result = new double[m.row][m.column];
			for(int row = 0;row<m.row;row++) {
				for(int column=0;column<m.column;column++) {
					result[row][column]=m.data[row][column];
				}
			}
		return result;
	}
	
	//display data
	public void print() {
		System.out.println("------------");
		for(int r=0;r<this.row;r++) {
			for(int c=0;c<this.column;c++) {
				System.out.print(this.data[r][c]+" ");
			}
			System.out.println();
		}
		System.out.println("------------");
	}
}

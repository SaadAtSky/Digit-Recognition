package MLP;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class StaticFunctions {
	static int[] labels1 = new int[2810];
	static int[] labels2 = new int[2810];
	static int[] labels3 = new int[50];
	static int[] labels4 = new int[50];
	
	//read data from files
	public static double[][] readFile1(File file){
		double[][] result = new double[2810][64];
		try {
			Scanner reader = new Scanner(file);
			for(int row = 0;row<2810;row++) {
				for(int column = 0;column < 65;column++) {
					if(column < 64) {
						int x = reader.nextInt();
						double normalized = (double)(x-0)/(16-0);
						result[row][column] = normalized;
					}
					else {
						labels1[row] = reader.nextInt();
					}
				}
			}
			reader.close();
		}
		catch(FileNotFoundException e) {
			System.out.println("An Error Occured");
			e.printStackTrace();
		}	
	return result;
	}
	//read data from files
	public static double[][] readFile2(File file){
		double[][] result = new double[2810][64];
		try {
			Scanner reader = new Scanner(file);
			for(int row = 0;row<2810;row++) {
				for(int column = 0;column < 65;column++) {
					if(column < 64) {
						int x = reader.nextInt();
						double normalized = (double)(x-0)/(16-0);
						result[row][column] = normalized;
					}
					else {
						labels2[row] = reader.nextInt();
					}
				}
			}
			reader.close();
		}
		catch(FileNotFoundException e) {
			System.out.println("An Error Occured");
			e.printStackTrace();
		}	
	return result;
	}
	
	//read data from files
	public static double[][] readFile3(File file){
		double[][] result = new double[50][64];
		try {
			Scanner reader = new Scanner(file);
			for(int row = 0;row<50;row++) {
				for(int column = 0;column < 65;column++) {
					if(column < 64) {
						int x = reader.nextInt();
						double normalized = (double)(x-0)/(16-0);
						result[row][column] = normalized;
					}
					else {
						labels3[row] = reader.nextInt();
					}
				}
			}
			reader.close();
		}
		catch(FileNotFoundException e) {
			System.out.println("An Error Occured");
			e.printStackTrace();
		}	
	return result;
	}
	//read data from files
	public static double[][] readFile4(File file){
		double[][] result = new double[50][64];
		try {
			Scanner reader = new Scanner(file);
			for(int row = 0;row<50;row++) {
				for(int column = 0;column < 65;column++) {
					if(column < 64) {
						int x = reader.nextInt();
						double normalized = (double)(x-0)/(16-0);
						result[row][column] = normalized;
					}
					else {
						labels4[row] = reader.nextInt();
					}
				}
			}
			reader.close();
		}
		catch(FileNotFoundException e) {
			System.out.println("An Error Occured");
			e.printStackTrace();
		}	
	return result;
	}
	
	//make a single input matrix
	public static double[][] singleInput(double[][] array,int index) {
		double[][] result = new double[array[0].length][1];
		for(int row = 0;row<result.length;row++) {
			result[row][0] = array[index][row];
		}
		return result;
	}
	
	public static double[][] makeTargetArray(int target){
		double[][] result = new double[10][1];
		for(int row=0;row<10;row++) {
			if(row==target) {
				result[row][0] = 1;
			}
			else {
				result[row][0] = 0;
			}
		}
		return result;
	}
	public static double[][] makeTargetArrayDemo(int target){
		double[][] result = new double[4][1];
		for(int row=0;row<4;row++) {
			if(row==target) {
				result[row][0] = 1;
			}
			else {
				result[row][0] = 0;
			}
		}
		return result;
	}
}

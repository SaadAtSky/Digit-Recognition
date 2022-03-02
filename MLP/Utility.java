package MLP;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Scanner;

/*
 * Contains the basic utility functions that are used by all three algorithms such as
 * file reading and creating separate labels and input arrays
 * creating the input and output arrays which are given as training data for the MLP classifier
 * finding the most frequent element in an array needed for the voting in SVM classifiers
 */
public class Utility {
	private static int sizeOfDataset = 2810;
	private static int numberOfFeatures = 64;
	private static int numberOfColumnsInDataset = 65;
	private static int[] labels1 = new int[sizeOfDataset];
	private static int[] labels2 = new int[sizeOfDataset];

	// read data from files
	public static double[][] readFile(File file) {
		int[] labels = new int[sizeOfDataset];
		double[][] result = new double[sizeOfDataset][numberOfFeatures];
		try {
			Scanner reader = new Scanner(file);
			for (int row = 0; row < sizeOfDataset; row++) {
				for (int column = 0; column < numberOfColumnsInDataset; column++) {
					if (column < numberOfFeatures) {
						int x = reader.nextInt();
						double normalized = (double) (x - 0) / (16 - 0);
						result[row][column] = normalized;
					} else {
						labels[row] = reader.nextInt();
					}
				}
			}
			if (file.getName().equals("dataset1")) {
				setLabels1(labels);
			} else if (file.getName().equals("dataset2")) {
				setLabels2(labels);
			}
			reader.close();
		} catch (FileNotFoundException e) {
			System.out.println("An Error Occured");
			e.printStackTrace();
		}
		return result;
	}

	// make a single input matrix
	public static double[][] singleInput(double[][] array, int index) {
		double[][] result = new double[array[0].length][1];
		for (int row = 0; row < result.length; row++) {
			result[row][0] = array[index][row];
		}
		return result;
	}

	// create an array to represents the expected activations
	// for the output layer in MLP classifier
	public static double[][] makeTargetArray(int target) {
		double[][] result = new double[10][1];
		for (int row = 0; row < 10; row++) {
			if (row == target) {
				result[row][0] = 1;
			} else {
				result[row][0] = 0;
			}
		}
		return result;
	}

	// Java program to find the most frequent element
	// in an array
	// The code for the following function is implemented using the contributions by
	// 'Akash Singh.'
	public static int mostFrequent(int arr[]) {
		// Sort the array
		Arrays.sort(arr);

		int lengthOfArray = arr.length;
		// find the max frequency using linear
		// traversal
		int max_count = 1, result = arr[0];
		int curr_count = 1;
		for (int i = 1; i < lengthOfArray; i++) {
			if (arr[i] == arr[i - 1])
				curr_count++;
			else {
				if (curr_count > max_count) {
					max_count = curr_count;
					result = arr[i - 1];
				}
				curr_count = 1;
			}
		}
		// If last element is most frequent
		if (curr_count > max_count) {
			max_count = curr_count;
			result = arr[lengthOfArray - 1];
		}
		return result;
	}

	public static int[] getLabels1() {
		return labels1;
	}

	public static void setLabels1(int[] labels1) {
		Utility.labels1 = labels1;
	}

	public static int[] getLabels2() {
		return labels2;
	}

	public static void setLabels2(int[] labels2) {
		Utility.labels2 = labels2;
	}
}

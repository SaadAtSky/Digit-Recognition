package NearestNeighbours;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.lang.Math;

/*
 * Implementation of a multiclass classifier based on Nearest Neighbours algorithm
 * to attempt to accurately classify hand-written digits using the MNIST Data set
 */
public class Classifier {
	private File fileDataset1;
	private File fileDataset2;
	private int sizeOfDataset = 2810;
	private int numberOfColumnsInDataset = 65;
	private int numberOfFeatures = 64;
	private int[] predictions1 = new int[sizeOfDataset];
	private int[] predictions2 = new int[sizeOfDataset];
	private int[][] dataset1 = new int[sizeOfDataset][numberOfColumnsInDataset];
	private int[][] dataset2 = new int[sizeOfDataset][numberOfColumnsInDataset];

	// store the two data sets in file variables
	public Classifier(File f1, File f2) {
		fileDataset1 = f1;
		fileDataset2 = f2;
	}

	// read data from the files
	public void readFile() {
		try {
			Scanner readerDataset1 = new Scanner(fileDataset1);
			Scanner readerDataset2 = new Scanner(fileDataset2);
			for (int row = 0; row < sizeOfDataset; row++) {
				for (int column = 0; column < numberOfColumnsInDataset; column++) {
					dataset1[row][column] = readerDataset1.nextInt();
					dataset2[row][column] = readerDataset2.nextInt();
				}
			}
			readerDataset1.close();
			readerDataset2.close();
		} catch (FileNotFoundException error) {
			System.out.println("An Error Occured");
			error.printStackTrace();
		}
	}

	// calculate the Euclidean distance between two points
	double distanceCalculator(int[] point1, int[] point2) {
		double distance = 0;
		for (int index = 0; index < numberOfFeatures; index++) {
			distance = distance + (point1[index] - point2[index]) * (point1[index] - point2[index]);
		}
		return Math.sqrt(distance);
	}

	// Dataset1 as test data, Dataset2 as train data
	public void test1() {
		int counter = 0;
		int[] testSetValue = new int[numberOfFeatures];
		int[] trainSetValue = new int[numberOfFeatures];
		double closestValue, currentValue;
		// predict values for each row in the test data and store in predictions1 array
		while (counter < sizeOfDataset) {// iterate through rows in test data
			for (int index = 0; index < numberOfFeatures; index++) {
				testSetValue[index] = dataset1[counter][index];
			}
			for (int index = 0; index < numberOfFeatures; index++) {// iterate through rows in train data
				trainSetValue[index] = dataset2[0][index];
			}
			closestValue = distanceCalculator(testSetValue, trainSetValue);
			predictions1[counter] = dataset2[0][numberOfFeatures];
			// iterate through rows in train data to find the closest value to the test
			// value
			for (int row = 1; row < sizeOfDataset; row++) {
				for (int index = 0; index < numberOfFeatures; index++) {
					trainSetValue[index] = dataset2[row][index];
				}
				currentValue = distanceCalculator(testSetValue, trainSetValue);// distance between test value and the
																				// current train value
				// keep track of the closest value and use it as the prediction for the test
				// value
				if (currentValue < closestValue) {
					closestValue = currentValue;
					predictions1[counter] = dataset2[row][numberOfFeatures];
				}
			}
			counter++;
		}
	}

	// Dataset2 as test data, Dataset1 as train data
	public void test2() {
		int counter = 0;
		int[] testSetValue = new int[numberOfFeatures];
		int[] trainSetValue = new int[numberOfFeatures];
		double closestValue, currentValue;
		// predict values for each row in the test data and store in predictions2 array
		while (counter < sizeOfDataset) {
			for (int index = 0; index < numberOfFeatures; index++) {// iterate through rows in test data
				testSetValue[index] = dataset2[counter][index];
			}
			for (int index = 0; index < numberOfFeatures; index++) {// iterate through rows in train data
				trainSetValue[index] = dataset1[0][index];
			}
			closestValue = distanceCalculator(testSetValue, trainSetValue);
			predictions2[counter] = dataset1[0][numberOfFeatures];
			// iterate through rows in train data to find the closest value to the test
			// value
			for (int row = 1; row < sizeOfDataset; row++) {
				for (int index = 0; index < numberOfFeatures; index++) {
					trainSetValue[index] = dataset1[row][index];
				}
				currentValue = distanceCalculator(testSetValue, trainSetValue);// distance between test value and the
																				// current train value
				// keep track of the closest value and use it as the prediction for the test
				// value
				if (currentValue < closestValue) {
					closestValue = currentValue;
					predictions2[counter] = dataset1[row][numberOfFeatures];
				}
			}
			counter++;
		}
	}

	// accuracy % for both data sets and overall average accuracy using this
	// implementation of Nearest Neighbours Algorithm
	public void resultsAccuracy() {
		int numberOfDatasets = 2;
		int correctPredictionsDataset1 = 0;
		int correctPredictionsDataset2 = 0;
		double accuracyDataset1 = 0;
		double accuracyDataset2 = 0;
		for (int index = 0; index < sizeOfDataset; index++) {
			if (predictions1[index] == dataset1[index][numberOfFeatures]) {
				correctPredictionsDataset1++;
			}
			if (predictions2[index] == dataset2[index][numberOfFeatures]) {
				correctPredictionsDataset2++;
			}
		}
		accuracyDataset1 = ((double) correctPredictionsDataset1 / sizeOfDataset) * 100;
		accuracyDataset2 = ((double) correctPredictionsDataset2 / sizeOfDataset) * 100;
		System.out.println("Accuracy Dataset1: " + accuracyDataset1 + " %");
		System.out.println("Accuracy Dataset2: " + accuracyDataset2 + " %");
		System.out.println("Overall Accuracy: " + ((accuracyDataset1 + accuracyDataset2) / numberOfDatasets) + " %");
	}
}

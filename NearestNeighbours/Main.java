package NearestNeighbours;

import java.io.File;

/*
 * An implementation of a nearest neighbours algortihm
 * trained on one part of the MNIST data set and tested it on the other and vice versa
 * The results are displayed using the metric accuracy score which is the percentage of correct predictions over total predictions
 */
public class Main {
	private static int sizeOfDataset = 2810;
	private static int[] predictions1 = new int[sizeOfDataset];
	private static int[] predictions2 = new int[sizeOfDataset];
	
	public static void setPredictions(Classifier classifier) {
		predictions1 = classifier.getPredictions1();
		predictions2 = classifier.getPredictions2();
	}
	public int[] getPredictions1() {
		return predictions1;
	}
	public int[] getPredictions2() {
		return predictions2;
	}
	public static void NN_ApplicationRunner() {
		// import the two files containing the data sets
		final String file1 = "dataset1";
		final String file2 = "dataset2";

		Classifier nearestNeighboursClassifier = new Classifier(new File(file1), new File(file2));
		nearestNeighboursClassifier.readFile();// read data from files
		nearestNeighboursClassifier.test1();// use data set 1 as test data
		nearestNeighboursClassifier.test2();// use data set 2 as test data
		nearestNeighboursClassifier.resultsAccuracy();// calculate and display the results
		
		setPredictions(nearestNeighboursClassifier);// store results in predictions arrays
	}
	public static void main(String[] args) {
		NN_ApplicationRunner();
	}
}

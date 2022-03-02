package NearestNeighbours;

import java.io.File;

/*
 * An implementation of a nearest neighbours algortihm
 * trained on one part of the MNIST data set and tested it on the other and vice versa
 * The results are displayed using the metric accuracy score which is the percentage of correct predictions over total predictions
 */
public class Main {

	public static void main(String[] args) {
		// import the two files containing the data sets
		File file1 = new File("dataset1");
		File file2 = new File("dataset2");

		Classifier nearestNeighboursClassifier = new Classifier(file1, file2);
		nearestNeighboursClassifier.readFile();// read data from files
		nearestNeighboursClassifier.test1();// use data set 1 as test data
		nearestNeighboursClassifier.test2();// use data set 2 as test data
		nearestNeighboursClassifier.resultsAccuracy();// calculate and display the results
	}
}

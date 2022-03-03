package MLP;

import java.io.File;

/*
 * Trains the classifier on one part of the MNIST data set and tests it on the other and vice versa
 * A multi-layer perceptron classifier is used with an input, multiple hidden layers, and an output layer
 * The results are displayed using the metric accuracy score which is the percentage of correct predictions over total predictions
 */
public class Main {
	private static int sizeOfDataset=2810;
	private static double[][] trainData,testData;
	private static int[] trainLabels,testLabels;
	private static int[] predictions = new int[sizeOfDataset];
	private static Classifier mlp;
	
	public int[] getPredictions() {
		return predictions;
	}
	// make predictions using the classifier. Then, calculate and display accuracy score
	static void predict() {
		int predictionIndex = 0;
		// count correct predictions for the test data
		while (predictionIndex < sizeOfDataset) {
			double[][] output = mlp.feedforward(Utility.singleInput(testData, predictionIndex));
			double highestValue = output[0][0];
			int prediction = 0;
			// iterate through the output nodes to keep track of the most activated node
			for (int outputNodesIndex = 0; outputNodesIndex < output.length; outputNodesIndex++) {
				if (output[outputNodesIndex][0] > highestValue) {
					highestValue = output[outputNodesIndex][0];
					prediction = outputNodesIndex;
				}
			}
			predictions[predictionIndex] = prediction;
			predictionIndex++;
		}
	}
	
	// display accuracy percentage and total correct predictions out of total predictions
	static void accuracy() {
		int correctPredictions = 0;
		for (int datasetIndex = 0; datasetIndex < sizeOfDataset; datasetIndex++) {
			if (testLabels[datasetIndex] == predictions[datasetIndex]) {
				correctPredictions++;
			}
		}

		System.out.println(correctPredictions+" out of "+sizeOfDataset);
		System.out.println("Accuracy: " + (double) correctPredictions / sizeOfDataset * 100);
	}
	public static void MLP_ApplicationRunner(File file1, File file2) {
		// define data and labels
		trainData = Utility.readFile(file1);
		testData = Utility.readFile(file2);
		if (file1.getName().equals("dataset1")) {
			trainLabels = Utility.getLabels1();
			testLabels = Utility.getLabels2();
		} else if (file1.getName().equals("dataset2")) {
			trainLabels = Utility.getLabels2();
			testLabels = Utility.getLabels1();
		}
		
		int inputNodes = 64;
		int hiddenNodes = 64;
		int outputNodes = 10;
		int numberOfHiddenLayers = 1;
		double[][] inputArray,targetArray;
		int epochsLimit = 20000;
		int sizeOfDataset = 2810;

		mlp = new Classifier(inputNodes, hiddenNodes, outputNodes, numberOfHiddenLayers);
		mlp.setLearningRate(0.1);

		// Batch training method implemented
		// with 'epochsLimit' representing the total number of times the classifier is trained on the Whole data set
		for (int epochs = 0; epochs < epochsLimit; epochs++) {
			// send inputs one by one to the classifier for training
			for (int datasetIndex = 0; datasetIndex < sizeOfDataset; datasetIndex++) {
				inputArray = Utility.singleInput(trainData, datasetIndex);
				targetArray = Utility.makeTargetArray(trainLabels[datasetIndex]);
				mlp.train(inputArray, targetArray);
			}
			// Changing of weights and biases once the whole data set is sent as input to the classifier for training
			mlp.change();
		}
		predict();
		
		accuracy();
	}
	public static void main(String[] args) {
		// set train and test files
		final String trainFile = "dataset1";
		final String testFile = "dataset2";

		MLP_ApplicationRunner(new File(trainFile),new File(testFile));

	}

}

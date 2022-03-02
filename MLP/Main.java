package MLP;

import java.io.File;

/*
 * Trains the classifier on one part of the MNIST data set and tests it on the other and vice versa
 * A multi-layer perceptron classifier is used with an input, multiple hidden layers, and an output layer
 * The results are displayed using the metric accuracy score which is the percentage of correct predictions over total predictions
 */
public class Main {
	public static void main(String[] args) {
		File file1 = new File("dataset1");
		File file2 = new File("dataset2");

		double[][] dataset1 = Utility.readFile(file1);
		double[][] dataset2 = Utility.readFile(file2);
		int[] testLabelsDataset1 = Utility.getLabels1();
		int[] testLabelsDataset2 = Utility.getLabels2();

		int inputNodes = 64;
		int hiddenNodes = 8;
		int outputNodes = 10;
		int numberOfHiddenLayers = 1;
		double[][] inputArray;
		double[][] targetArray;
		int epochsLimit = 50000;
		int sizeOfDataset = 2810;

		Classifier mlp = new Classifier(inputNodes, hiddenNodes, outputNodes, numberOfHiddenLayers);
		mlp.setLearningRate(0.1);

		System.out.println("Started Training");
		// Batch training method implemented
		// with 'epochsLimit' representing the total number of times the classifier is
		// trained on the Whole data set
		for (int epochs = 0; epochs < epochsLimit; epochs++) {
			if (epochs % 1000 == 0) {
				System.out.println(epochs);
			}
			// send inputs one by one to the classifier for training
			for (int datasetIndex = 0; datasetIndex < sizeOfDataset; datasetIndex++) {
				inputArray = Utility.singleInput(dataset1, datasetIndex);
				targetArray = Utility.makeTargetArray(testLabelsDataset1[datasetIndex]);
				mlp.train(inputArray, targetArray);
			}
			// Changing of weights and biases once the whole data set is sent as input to
			// the classifier for training
			mlp.change();
		}
		System.out.println("Finished Training");

		int predictionIndex = 0;
		int correctPredictions = 0;
		// count correct predictions for the test data
		while (predictionIndex < sizeOfDataset) {
			double[][] output = mlp.feedforward(Utility.singleInput(dataset2, predictionIndex));
			double highestValue = output[0][0];
			int prediction = 0;
			// iterate through the output nodes to keep track of the most activated node
			for (int outputNodesIndex = 0; outputNodesIndex < output.length; outputNodesIndex++) {
				if (output[outputNodesIndex][0] > highestValue) {
					highestValue = output[outputNodesIndex][0];
					prediction = outputNodesIndex;
				}
			}
			// increment 'correctPredictions' if the most activated node represents the
			// correct label
			if (prediction == testLabelsDataset2[predictionIndex]) {
				correctPredictions++;
			}
//			System.out.println("PREDICTION: "+prediction);
//			System.out.println("ANSWER: "+Utility.getLabels2()[predictionIndex]);
			predictionIndex++;
		}
		System.out.println("Accuracy: " + (double) correctPredictions / sizeOfDataset * 100);
	}

}

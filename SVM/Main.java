package SVM;

import java.io.File;

import MLP.Utility;

/*
 * Trains the classifier on one part of the MNIST data set and tests it on the other and vice versa
 * One SVM classifier is trained for each of the unique class pairs
 * and the predictions are generated using a voting system
 * The results are displayed using the metric accuracy score which is the percentage of correct predictions over total predictions
 */
public class Main {
	private static int sizeOfDataset = 2810;
	private static int numberOfClassPairs = 45;
	private static int pairOfClasses = 2;
	private static Classifier[] classifiers = new Classifier[numberOfClassPairs];
	private static int[] votingArray = new int[numberOfClassPairs];
	private static int[] predictions = new int[sizeOfDataset];
	private static double[][] dataset1, dataset2;
	private static int[] LabelsDataset1, LabelsDataset2;

	public static void trainSingleClassifier(int classifierCounter) {
		int epochsLimit = 50000;
		int patienceEpochs = 5000;// iterations interval to check if the cost function value is reducing
		// train the data until the EPOCHS limit or patience EPOCHS
		
		// store the cost function value to test for patience EPOCHS
		double oldCostFunction = classifiers[classifierCounter].computeCostFunction();
		for (int epochs = 0; epochs < epochsLimit; epochs++) {
//			if (epochs % 1000 == 0) {
//				System.out.println(epochs);
//			}
			// choose to either train using the batched or sochastic gradient descent method
			classifiers[classifierCounter].batchGradientDescent();
			if (epochs % patienceEpochs == 0 && epochs != 0) {
				if ((int) classifiers[classifierCounter].computeCostFunction() >= (int) oldCostFunction) {
//					classifiers[classifierCounter].dropLearningRate();
					System.out.println("EPOCHS: " + epochs);
					break;
				} else {
					oldCostFunction = classifiers[classifierCounter].computeCostFunction();
					System.out.println(oldCostFunction);
				}
			}
		}
	}

	// make predictions using each SVM classifier and store the most frequently
	// occured value as the predicted value for that particular input data
	public static void predictions() {
		for (int datasetIndex = 0; datasetIndex < sizeOfDataset; datasetIndex++) {
			for (int classifierIndex = 0; classifierIndex < numberOfClassPairs; classifierIndex++) {
				votingArray[classifierIndex] = classifiers[classifierIndex].predict(dataset2[datasetIndex]);
			}
			// choose the most frequent answer as the prediction
			predictions[datasetIndex] = Utility.mostFrequent(votingArray);
		}
	}

	// calculate and display accuracy score
	public static void accuracy() {
		int correctPredictions = 0;
		for (int datasetIndex = 0; datasetIndex < sizeOfDataset; datasetIndex++) {
			if (LabelsDataset2[datasetIndex] == predictions[datasetIndex]) {
				correctPredictions++;
			}
		}

		// display accuracy percentage and total correct predictions out of total predictions
		System.out.println(correctPredictions+" out of "+sizeOfDataset);
		System.out.println("Accuracy: " + (double) correctPredictions / sizeOfDataset * 100);
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		File file1 = new File("dataset1");
		File file2 = new File("dataset2");

		// define data and labels
		dataset1 = Utility.readFile(file1);
		dataset2 = Utility.readFile(file2);
		LabelsDataset1 = Utility.getLabels1();
		LabelsDataset2 = Utility.getLabels2();

		int classifierCounter = 0;
		int secondClassFirstElement = 1;
		int highestValueFirstClass = 9;
		int highestValueSecondClass = 10;
		int[] classes = new int[pairOfClasses];

		// train a classifer for each of the unique class pairs starting from 0/1 and
		// ending at 8/9
		for (int firstClass = 0; firstClass < highestValueFirstClass; firstClass++) {
			for (int secondClass = secondClassFirstElement; secondClass < highestValueSecondClass; secondClass++) {
				// set the two classes
				classes[0] = firstClass;
				classes[1] = secondClass;

				// instantiate and train a new SVM classifier for each unique pair
				classifiers[classifierCounter] = new Classifier(dataset1, LabelsDataset1, classes);
				trainSingleClassifier(classifierCounter);
				
				classifierCounter++;
				System.out.println("------------ " + firstClass + "/" + secondClass + " -------------");
				System.out.println("Classifier Counter: " + classifierCounter);// max 44
			}
			secondClassFirstElement++;
		}

		predictions();

		accuracy();

	}

}

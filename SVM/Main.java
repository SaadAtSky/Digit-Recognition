package SVM;

import java.io.File;

import MLP.StaticFunctions;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		File file1 = new File("dataset1");
		File file2 = new File("dataset2");

		int sizeOfDataset = 2810;
		int numberOfClassPairs = 45;
		int pairOfClasses = 2;
		int[] classes = new int[pairOfClasses];
		SVM_algorithm[] classifiers = new SVM_algorithm[numberOfClassPairs];
		int[] votingArray = new int[numberOfClassPairs];
		
		//train and test dataset
		double[][] dataset1 = StaticFunctions.readFile1(file1);
		double[][] dataset2 = StaticFunctions.readFile2(file2);
		int[] testLabelsDataset1 = StaticFunctions.getLabels1();
		int[] testLabelsDataset2 = StaticFunctions.getLabels2();
		
		int[] predictions = new int[sizeOfDataset];
		int classifierCounter=0;
		int secondClassFirstElement = 1;
		double oldCostFunction;
		int epochsLimit = 1000;

		for (int firstClass = 0; firstClass < 9; firstClass++) {
			for (int secondClass = secondClassFirstElement; secondClass < 10; secondClass++) {
				// set the two classes
				classes[0] = firstClass;
				classes[1] = secondClass;

				classifiers[classifierCounter] = new SVM_algorithm(dataset2, StaticFunctions.getLabels2(), classes);
				// train
				oldCostFunction = classifiers[classifierCounter].computeCostFunction();
				for (int epochs = 0; epochs < epochsLimit; epochs++) {
//					if (epochs % 1000 == 0) {
//						System.out.println(epochs);
//					}
					classifiers[classifierCounter].train();
					if (epochs % 100 == 0 && epochs != 0) {
						if ((int) classifiers[classifierCounter].computeCostFunction() >= (int) oldCostFunction) {
//							System.out.println(epochs);
							break;
						} else {
							oldCostFunction = classifiers[classifierCounter].computeCostFunction();
//							System.out.println(oldCostFunction);
						}
					}
				}			
				classifierCounter++;
				System.out.println("------------ "+firstClass+"/"+secondClass+" -------------");
				System.out.println("Classifier Counter: " + classifierCounter);//max 44
			}
			secondClassFirstElement++;
		}
		
		//make predictions and send to check accuracy		
		for(int datasetIndex=0;datasetIndex<sizeOfDataset;datasetIndex++) {
			for(int classifierIndex = 0; classifierIndex < numberOfClassPairs;classifierIndex++) {
				votingArray[classifierIndex]=classifiers[classifierIndex].predict(dataset1[datasetIndex]);
			}
			//perform voting = choose the most preferrent answer
			predictions[datasetIndex] = StaticFunctions.mostFrequent(votingArray);
		}
		
//		System.out.println("Predictions");
//		for(int index = 0;index<50;index++) {
//			System.out.println(predictions[index]);
//		}
		
		//check accuracy
		int correctPredictions = 0;
		for(int datasetIndex = 0;datasetIndex<sizeOfDataset;datasetIndex++) {
			if(testLabelsDataset1[datasetIndex]==predictions[datasetIndex]) {
				correctPredictions++;
			}
		}
		
		//display accuracy percentage
		System.out.println("Correct Predictions: " + correctPredictions);
		System.out.println("Accuracy: " + (double)correctPredictions/sizeOfDataset*100);
	}

}

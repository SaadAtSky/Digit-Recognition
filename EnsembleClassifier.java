import java.io.File;
import MLP.Utility;

public class EnsembleClassifier {
	private static int sizeOfDataset = 2810;
	private static int[] ensemblePredictions = new int[sizeOfDataset];
	
	// calculate and display accuracy score
	public static void accuracy(File testFile) {
		int correctPredictions = 0;
		int[] testLabels = new int[sizeOfDataset];
		if (testFile.getName().equals("dataset1")) {
			testLabels = Utility.getLabels1();
		} else if (testFile.getName().equals("dataset2")) {
			testLabels = Utility.getLabels2();
		}
		for (int datasetIndex = 0; datasetIndex < sizeOfDataset; datasetIndex++) {
			if (testLabels[datasetIndex] == ensemblePredictions[datasetIndex]) {
				correctPredictions++;
			}
		}

		// display accuracy percentage and total correct predictions out of total predictions
		System.out.println(correctPredictions+" out of "+sizeOfDataset);
		System.out.println("Accuracy: " + (double) correctPredictions / sizeOfDataset * 100);
	}

	public static void main(String[] args) {
		int numberOfClassifiers = 3;
		int[] votingArray = new int[numberOfClassifiers];
		NearestNeighbours.Main NN = new NearestNeighbours.Main();
		MLP.Main MLP = new MLP.Main();
		SVM.Main SVM = new SVM.Main();
		
		// set train and test files
		final String trainFile = "dataset1";
		final String testFile = "dataset2";
		
		// Nearest neighbours predictions
		NN.NN_ApplicationRunner();
		int[] NN_Predictions = NN.getPredictions2();

		// Multilayer Perceptron predictions
		MLP.MLP_ApplicationRunner(new File(trainFile),new File (testFile));
		int[] MLP_Predictions = MLP.getPredictions();
		
		// Support vector machines predictions
		SVM.SVM_ApplicationRunner(new File(trainFile),new File(testFile));
		int[] SVM_Predictions = SVM.getPredictions();
		
		// iterate through the data set and build the ensemble predictions array
		for(int datasetIndex = 0;datasetIndex<sizeOfDataset;datasetIndex++) {
			//build the voting array using the predictions from each classfier
			votingArray[0] = NN_Predictions[datasetIndex];
			votingArray[1] = MLP_Predictions[datasetIndex];
			votingArray[2] = SVM_Predictions[datasetIndex];
			// if all three predictions are different, then go with NN predictions as it has highest accuracy
			if(votingArray[0]!=votingArray[1] && votingArray[0]!=votingArray[2] && votingArray[1]!= votingArray[2]) {
				ensemblePredictions[datasetIndex] = NN_Predictions[datasetIndex];
			}
			else {
				ensemblePredictions[datasetIndex] = Utility.mostFrequent(votingArray);
			}
		}
		accuracy(new File(testFile));
		
	}

}

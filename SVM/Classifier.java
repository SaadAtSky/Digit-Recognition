package SVM;

import MLP.Matrix;

/*
 * Implementation of a multiclass classifier based on Support Vector Machines
 * to attempt to accurately classify hand-written digits using the MNIST Data set
 */
public class Classifier {
	private Matrix weights;
	private Matrix trainData;
	private Matrix labels;
	private int[] classes;
	private double marginConstant = 1000;
	private double learningRate = 0.0001;

	Classifier(double[][] trainDataArray, int[] labelsArray, int[] classesArray) {
		// set the unique pair of classes to be classified
		setClasses(java.util.Arrays.copyOf(classesArray, classesArray.length));

		// initialize weights
		weights = new Matrix(trainDataArray[0].length + 1, 1);
		randomize(weights);

		// initialize trainData and labels
		filterData(trainDataArray, labelsArray);
		trainData = Matrix.addOnes(trainData);

		// setLabels
		setLabels();
	}

	// make prediction (1 or -1) for a single input value
	// based on if the point if above or below the hyper plane
	// and return the associated class value
	public int predict(double[] trainDataArray) {
		// w.x + b >= 1 or <=-1
		Matrix testData = Matrix.transpose(Matrix.from1DArrayDouble(trainDataArray));
		testData = Matrix.addOnes(testData);
		Matrix result = new Matrix(1, 1);

		result = Matrix.dotProduct(testData, weights);
		result = positionOfPoint(result);

		if (result.getData()[0][0] == 1) {
			return this.getClasses()[0];
		} else {
			return this.getClasses()[1];
		}
	}

	// adjust the wieghts after calculating iterating through the
	// whole batch of data and calculating the average gradient
	public void batchGradientDescent() {
		double numberOfSamples = trainData.getRow();
		Matrix result = new Matrix(weights.getRow(), weights.getColumn());
		Matrix predictions = Matrix.dotProduct(trainData, weights);
		for (int row = 0; row < predictions.getRow(); row++) {
			double hingeLoss = max(1 - (labels.getData()[row][0] * predictions.getData()[row][0]));
			if (hingeLoss == 0) {
				result = Matrix.add(result, weights);
			} else {
				// delta = weights - Constant*labels*input
				Matrix delta = new Matrix(weights.getRow(), weights.getColumn());
				for (int index = 0; index < weights.getRow(); index++) {
					delta.getData()[index][0] = trainData.getData()[row][index] * marginConstant
							* labels.getData()[row][0];
				}
				delta = Matrix.subtract(weights, delta);
				result = Matrix.add(result, delta);
			}
		}
		Matrix gradient = Matrix.scalarMultiply(result, (1 / numberOfSamples));

		adjustWeights(gradient);
	}

	// adjust the weights with each row of the input data
	public void sochasticGradientDescent() {
		Matrix result = new Matrix(weights.getRow(), weights.getColumn());
		Matrix predictions = Matrix.dotProduct(trainData, weights);
		for (int row = 0; row < predictions.getRow(); row++) {
			double hingeLoss = max(1 - (labels.getData()[row][0] * predictions.getData()[row][0]));
			if (hingeLoss == 0) {
				result = weights;
			} else {
				// delta = w - Cyx
				result = new Matrix(weights.getRow(), weights.getColumn());
				for (int index = 0; index < weights.getRow(); index++) {
					result.getData()[index][0] = trainData.getData()[row][index] * marginConstant
							* labels.getData()[row][0];
				}
				result = Matrix.subtract(weights, result);
			}
			adjustWeights(result);
		}
	}

	public double computeCostFunction() {
		double value, hingeLoss;
		double numberOfSamples = trainData.getRow();
		double weightMagnitude = Matrix.absoluteValue(weights);
		Matrix prediction = Matrix.dotProduct(trainData, weights);
		double sum = 0;
		for (int r = 0; r < labels.getRow(); r++) {
			hingeLoss = max(1 - labels.getData()[r][0] * prediction.getData()[r][0]);
			sum = sum + hingeLoss;
		}
		value = ((Math.pow(weightMagnitude, 2) / 2) + marginConstant * ((1 / numberOfSamples) * sum));
		return value;
	}

	void adjustWeights(Matrix gradient) {
		weights = Matrix.subtract(weights, Matrix.scalarMultiply(gradient, learningRate));

	}

	void randomize(Matrix matrix) {
		for (int row = 0; row < matrix.getRow(); row++) {
			for (int column = 0; column < matrix.getColumn(); column++) {
				matrix.getData()[row][column] = 0;
			}
		}
	}

	// set labels as 1 or -1 based on the unique class pair for this instance of the classifier
	// e.g for class (0,2) all 0's will be labelled as 1 and 2's as -1
	void setLabels() {
		// iterate throught labels
		for (int labelIndex = 0; labelIndex < labels.getRow(); labelIndex++) {
			if (labels.getData()[labelIndex][0] == getClasses()[0]) {
				labels.getData()[labelIndex][0] = 1;
			} else if (labels.getData()[labelIndex][0] == getClasses()[1]) {
				labels.getData()[labelIndex][0] = -1;
			}
		}
	}

	// create a subset of the data set to only include those data points that
	// have the labels according to the class pair for this instance of the classifier
	void filterData(double[][] originalDataset, int[] originalLabels) {
		int datasetSize = 2810;
		int numberOfFeatures = 64;
		int trainDataSize = findTrainDataSize(originalDataset, originalLabels);
		boolean found = false;
		int dataCounter = 0;
		double[][] subsetTrainData = new double[trainDataSize][numberOfFeatures];
		int[] subsetLabels = new int[trainDataSize];

		for (int labelIndex = 0; labelIndex < datasetSize; labelIndex++) {
			found = false;
			for (int classIndex = 0; classIndex < getClasses().length; classIndex++) {
				if (originalLabels[labelIndex] == getClasses()[classIndex]) {
					found = true;
				}
			}
			if (found) {
				for (int index = 0; index < numberOfFeatures; index++) {
					subsetTrainData[dataCounter][index] = originalDataset[labelIndex][index];
				}
				subsetLabels[dataCounter] = originalLabels[labelIndex];
				dataCounter++;
			}
		}
		trainData = Matrix.fromArray(subsetTrainData);
		labels = Matrix.from1DArray(subsetLabels);
	}

	// find the number of data points to be included in the subset of the dataset
	// based on the class pair
	int findTrainDataSize(double[][] originalDataset, int[] originalLabels) {
		int trainDataSize = 0;
		int datasetSize = 2810;
		boolean found = false;
		for (int labelIndex = 0; labelIndex < datasetSize; labelIndex++) {
			for (int classIndex = 0; classIndex < getClasses().length; classIndex++) {
				if (originalLabels[labelIndex] == getClasses()[classIndex]) {
					found = true;
				}
			}
			if (found) {
				trainDataSize++;
			}
			found = false;
		}
		return trainDataSize;
	}

	// find the maximum value with between 0 and the input value
	double max(double value) {
		if (value > 0) {
			return value;
		} else {
			return 0;
		}
	}

	// assign 1 or -1 representing if the point is above or below the hyperplane
	Matrix positionOfPoint(Matrix matrix) {
		Matrix result = new Matrix(matrix.getRow(), matrix.getColumn());
		for (int row = 0; row < matrix.getRow(); row++) {
			if (matrix.getData()[row][0] > 0) {
				result.getData()[row][0] = 1;
			} else if (matrix.getData()[row][0] < 0) {
				result.getData()[row][0] = -1;
			}
		}
		return result;
	}

	public int[] getClasses() {
		return classes;
	}

	void setClasses(int[] classes) {
		this.classes = classes;
	}

	public void dropLearningRate() {
		this.learningRate = getLearningRate() / 2;
	}

	public double getLearningRate() {
		return learningRate;
	}
}

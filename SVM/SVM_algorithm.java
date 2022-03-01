package SVM;

import MLP.Matrix;

public class SVM_algorithm {
	Matrix weights;
	Matrix trainData;
	Matrix labels;
	private int[] classes;
	double marginConstant = 5000;
	double learningRate = 0.00001;
	
	SVM_algorithm(double[][] td, int[] l, int[] c) {
		//classes
		setClasses(java.util.Arrays.copyOf(c,c.length));
		
		//initialize weights
		weights = new Matrix(td[0].length+1,1);
		randomize(weights);
//		System.out.println("Weights: ");
//		weights.print();
		
		//initialize trainData and labels
		filterData(td,l);
		trainData = Matrix.addOnes(trainData);
		
		//setLabels
		setLabels();
		
//		System.out.println("Train Data: ");
//		trainData.print();
		
//		System.out.println("Labels: ");
//		labels.print();
	}
	
	public int predict(double[] td) {
		//w.x + b >= 1 or <=-1
		Matrix testData = Matrix.transpose(Matrix.from1DArrayDouble(td));
		testData = Matrix.addOnes(testData);
		Matrix result = new Matrix(1,1);
//		testData.print();
		
		result = Matrix.multiply(testData,weights);
		result = Matrix.SVM_activation(result);

//		for(int i =0;i<classes.length;i++) {
//			System.out.println(classes[i]);
//		}

		if(result.getData()[0][0]==1) {
			return this.getClasses()[0];
		}
		else {
			return this.getClasses()[1];
		}
	}
	public void train() {
		
		batch();		
	
//		SGD();
		
	}
	public void batch() {
		double numberOfSamples = trainData.getRow();
		
		Matrix gradient = Matrix.scalarMultiply(Matrix.multiplyMaximumGradient(labels, trainData,weights,marginConstant),(1/numberOfSamples));
		
//		System.out.println("Gradient: ");
//		gradient.print();
		
		adjustWeights(gradient);
	}
	public void SGD() {
		Matrix result = new Matrix(weights.getRow(),weights.getColumn());
		Matrix predictions = Matrix.multiply(trainData , weights);
		for (int row = 0; row < predictions.getRow(); row++) {
			double hingeLoss = Matrix.max(1 - (labels.getData()[row][0] * predictions.getData()[row][0]));
			if(hingeLoss == 0) {
				result = weights;
			}
			else {
				//delta = w - Cyx
				result = new Matrix(weights.getRow(),weights.getColumn());
				for(int index = 0;index<weights.getRow();index++) {
					result.getData()[index][0] = trainData.getData()[row][index]*marginConstant*labels.getData()[row][0];
				}
				result = Matrix.subtract(weights, result);
			}
			adjustWeights(result);
		}
	}
	
	public double computeCostFunction() {
		double value;
		double numberOfSamples = trainData.getRow();
		double weightMagnitude = Matrix.magnitude(weights);
		Matrix prediction = Matrix.multiply(trainData , weights);
		
		value = ((Math.pow(weightMagnitude, 2)/2) + marginConstant * ( (1/numberOfSamples)*Matrix.multiplyMaximum(labels, prediction) ));
		
//		System.out.println("Cost Function: " + value);
		
		return value;
		
	}
	public void adjustWeights(Matrix gradient) {
		//newW = oldW - LR*Gradient
		weights = Matrix.subtract(weights, Matrix.scalarMultiply(gradient, learningRate));
		
//		System.out.println("New Weights: ");
//		weights.print();
		
	}
	public void randomize(Matrix m) {
		for(int row = 0; row<m.getRow();row++) {
			for(int column=0;column<m.getColumn();column++) {
//				m.getData()[row][column] = (double)Math.random()*2-1;//value between 1 and -1
				m.getData()[row][column] = 0;//
			}
		}
	}
	public void setLabels() {
		//iterate throught labels
//		System.out.println("Labels: ");
//		labels.print();
		for(int labelIndex = 0;labelIndex<labels.getRow();labelIndex++) {
			if(labels.getData()[labelIndex][0] == getClasses()[0]) {
				labels.getData()[labelIndex][0]=1;
			}
			else if(labels.getData()[labelIndex][0] == getClasses()[1]){
				labels.getData()[labelIndex][0]=-1;
			}
		}
	}
	public void filterData(double[][] originalDataset,int[] originalLabels) {// produce new labels and data
		int datasetSize = 2810;
		int arraySize = 0;
		boolean found = false;
		int dataCounter=0;
		//go through all labels and see if classification(loop) == label
		for(int labelIndex = 0;labelIndex<datasetSize;labelIndex++) {
			for(int classIndex =0;classIndex < getClasses().length;classIndex++) {
				if(originalLabels[labelIndex] == getClasses()[classIndex]) {
					found = true;
				}
			}
			if(found) {
				arraySize++;
			}
			found = false;
		}
//		System.out.println("Array size "+arraySize);
		double[][] subsetTrainData = new double[arraySize][64];
		int[] subsetLabels = new int[arraySize];
		
		for(int labelIndex = 0;labelIndex<datasetSize;labelIndex++) {
			found = false;
			for(int classIndex =0;classIndex < getClasses().length;classIndex++) {
				if(originalLabels[labelIndex] == getClasses()[classIndex]) {
					found = true;
				}
			}
			if(found) {
				for(int index = 0;index<64;index++) {
					subsetTrainData[dataCounter][index] = originalDataset[labelIndex][index];
				}
				subsetLabels[dataCounter] = originalLabels[labelIndex];
				dataCounter++;
			}
		}
		trainData = Matrix.fromArray(subsetTrainData);
		labels = Matrix.from1DArray(subsetLabels);
	}
//	public void checkAccuracy(double[][] testData, int[] testLabels,int[] classifications) {
//		int labelsInDataset = 0;
//		int correctPredictions = 0;
//		boolean found = false;
//		
//		Matrix predictions = predict(testData);
//		labels = Matrix.from1DArray(testLabels);
//		setLabels();
//		
//		System.out.println("Predictions: ");
//		predictions.print();
//
//		for(int row = 0;row<labels.getRow();row++) {
//			if(labels.getData()[row][0]==predictions.getData()[row][0]) {
//				correctPredictions++;
//			}
//		}
//		
//		System.out.println("New Weights: ");
//		weights.print();
//		
//		//display accuracy percentage
//		System.out.println("Correct Predictions: " + correctPredictions);
//		System.out.println("Accuracy: " + (double)correctPredictions/2810*100);
//	}

	public int[] getClasses() {
		return classes;
	}

	public void setClasses(int[] classes) {
		this.classes = classes;
	}
}

package MLP;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class MultiLayerPerceptron {
	Matrix inputNodes,hiddenNodes,outputNodes;
	Matrix hiddenWeights,outputWeights;
	Matrix hiddenBias,outputBias;
	Matrix outputWeightsDelta,hiddenWeightsDelta;
	Matrix outputGradient,hiddenGradient;
	Matrix averageError;
	
	double learningRate = 0.1;
	
	public MultiLayerPerceptron(int in,int hn, int on) {
		//initialize the input,hidden and output nodes/neurons
		inputNodes = new Matrix(in,1);
		hiddenNodes = new Matrix(hn,1);
		outputNodes = new Matrix(on,1);
		
		averageError = new Matrix(on,1);
		
		outputGradient = new Matrix(on,1);
		hiddenGradient = new Matrix(on,1);
		
		hiddenWeightsDelta = new Matrix(hn,in);
		outputWeightsDelta = new Matrix(on,hn);
		
		//initialize the weights for hidden output layer
		hiddenWeights = new Matrix(hn,in);
		outputWeights = new Matrix(on,hn);
		randomize(hiddenWeights);
		randomize(outputWeights);
		
		//initialize bias for hidden and output layer
		hiddenBias = new Matrix(hn,1);
		outputBias = new Matrix(on,1);
		setBias(hiddenBias);
		setBias(outputBias);
	}
	
	public void randomize(Matrix m) {
		for(int row = 0; row<m.row;row++) {
			for(int column=0;column<m.column;column++) {
				m.data[row][column] = (double)Math.random()*2-1;//value between 1 and -1
			}
		}
	}
	
	public void setBias(Matrix m) {
		for(int row = 0; row<m.row;row++) {
			for(int column=0;column<m.column;column++) {
				m.data[row][column] = (double)Math.random()*2-1;
			}
		}
	}
	
	
	public double[][] feedforward(double[][] inputs) {
		double[][] result;
		
		//initialize inputNodes
		inputNodes = Matrix.fromArray(inputs);
//		System.out.println("INPUT");
//		inputNodes.print();
		
//		for(int index = 0; index<2;index++) {
			//multiply inputs with hidden weights to get hidden layer
			hiddenNodes = Matrix.multiply(hiddenWeights, inputNodes);
			
			//add bias to result
//			hiddenNodes = Matrix.add(hiddenNodes, hiddenBias);
//			System.out.println("BIAS");
//			hiddenNodes.print();
			
			//apply the sigmoid activation function
			hiddenNodes = Matrix.activation(hiddenNodes);
//		}
		//multiply hidden layer with output weights to get output layer
		outputNodes = Matrix.multiply(outputWeights, hiddenNodes);
		
		//add bias to result
//		outputNodes = Matrix.add(outputNodes, outputBias);
		//apply the sigmoid activation function
		outputNodes = Matrix.activation(outputNodes);
		
		//store result value
		result = Matrix.toArray(outputNodes);
		
		return result;
	}
	
	public void train(double[][]inputs, double[][] targetArray){
		//get results
		double[][] resultArray = feedforward(inputs);
		Matrix result = Matrix.fromArray(resultArray);
		Matrix target = Matrix.fromArray(targetArray);
////		result.print();
////		target.print();
//		
//		//calculate output error = output - target
		Matrix outputError = Matrix.subtract(target, result);
		averageError = Matrix.add(averageError, outputError);
//		System.out.println("OUTPUT ERROR");
//		outputError.print();
		
		//transpose hidden weights
		Matrix transposedOutputWeights = Matrix.transpose(outputWeights);
//		outputWeights.print();
//		System.out.println("TRANSPOSED OUTPUT WEIGHTS");
//		transposedOutputWeights.print();
//		
		//calculate hidden error = transposed output weights * output error
		Matrix hiddenError = Matrix.multiply(transposedOutputWeights, outputError);
//		System.out.println("HIDDEN ERROR");
//		hiddenError.print();
		
		//hidden gradient
		outputGradient = Matrix.deactivate(result);
//		System.out.println("HIDDEN GRADIENT");
//		gradient.print();
	
		//multiply error
		outputGradient = Matrix.elementWise(outputError, outputGradient);
		
		//multiply Learning Rate
		outputGradient = Matrix.scalarMultiply(outputGradient, learningRate);
		
		//multiply transposed weights
		outputWeightsDelta = Matrix.add(outputWeightsDelta,Matrix.multiply(outputGradient, Matrix.transpose(hiddenNodes)));
		
		//adjust bias
//		outputBias = Matrix.add(outputBias, gradient);
//		System.out.println("OUTPUT BIAS");
//		outputBias.print();
		
		//new weight = original weight + delta weight(transposed back to original)
//		System.out.println("OUTPUT WEIGHTS BEFORE");
//		outputWeights.print();
//		outputWeights = Matrix.add(outputWeights, outputWeightsDelta);
//		System.out.println("OUTPUT WEIGHTS AFTER");
//		outputWeights.print();
		
		//gradient input
		hiddenGradient = Matrix.deactivate(hiddenNodes);
		
		//multiple error
		hiddenGradient = Matrix.elementWise(hiddenError, hiddenGradient);
		
		//multiply lr
		hiddenGradient = Matrix.scalarMultiply(hiddenGradient, learningRate);
		
		//adjust bias
//		hiddenBias = Matrix.add(hiddenBias, gradient);
		
		//multiply transposed weights
		hiddenWeightsDelta = Matrix.add(hiddenWeightsDelta,Matrix.multiply(hiddenGradient, Matrix.transpose(inputNodes)));
		
		//add original wieght
//		System.out.println("HIDDEN WEIGHTS BEFORE");
//		hiddenWeights.print();
//		hiddenWeights = Matrix.add(hiddenWeights, hiddenWeightsDelta);
//		System.out.println("HIDDEN WEIGHTS AFTER");
//		hiddenWeights.print();
	}
	
	public void change() {
		//final values
		outputWeightsDelta = Matrix.scalarMultiply(outputWeightsDelta, (double)1/2810);
		hiddenWeightsDelta = Matrix.scalarMultiply(hiddenWeightsDelta, (double)1/2810);
		averageError = Matrix.scalarMultiply(averageError, (double)1/2810);
//		averageError.print();
		
		outputWeights = Matrix.add(outputWeights, outputWeightsDelta);
//		outputWeights.print();
//		outputBias = Matrix.add(outputBias, outputGradient);
		hiddenWeights = Matrix.add(hiddenWeights, hiddenWeightsDelta);
//		hiddenBias = Matrix.add(hiddenBias, hiddenGradient);
				
		//display cost function
//		System.out.println("Cost Function: "+Matrix.costFunction(averageError));
		
		averageError = new Matrix(averageError.row,1);
		outputGradient = new Matrix(outputGradient.row,1);
		hiddenGradient = new Matrix(hiddenGradient.row,1);
		hiddenWeightsDelta = new Matrix(hiddenWeightsDelta.row,hiddenWeightsDelta.column);
		outputWeightsDelta = new Matrix(outputWeightsDelta.row,outputWeightsDelta.column);
	}
	
	

	
//	public void test() {
//		double sum;
//		int counter = 0;
//		while(counter < 1) {
//			sum = 0;
//			for(int index=0;index<2;index++) {
//				sum = sum + inputs[counter][index]*weights[counter][index];
//			}
//			predictions[counter] = sigmoid(sum);
//			counter++;
//		}
//	}
	
//	public void accuracy() {
//		int correctPredictions = 0;
//		double accuracy;
//		for(int index = 0;index<inputs.length;index++) {
//			if(predictions[index]==labels[index]) {
//				correctPredictions++;
//			}
//		}
//		accuracy = (double)correctPredictions/inputs.length;
//		System.out.println("Accuracy: "+accuracy*100+" %");
//	}
//	
//	public void printInputs() {
//		for(int row = 0;row<1;row++) {
//			for(int column = 0;column < 2;column++) {
//				System.out.print(inputs[row][column]+", ");
//			}
//			System.out.println();
//		}
//	}
	
//	public void printWeights() {
//		for(int row = 0;row<1;row++) {
//			for(int column = 0;column < 2;column++) {
//				System.out.print(weights[row][column]+", ");
//			}
//			System.out.println();
//		}
//	}
//	
//	public void printLabels() {
//		for(int row = 0;row<1;row++) {
//			System.out.print(labels[row]+", ");
//		}
//	}
//	public void printPredictions() {
//		for(int row = 0;row<1;row++) {
//			System.out.print(predictions[row]+", ");
//		}
//	}
}

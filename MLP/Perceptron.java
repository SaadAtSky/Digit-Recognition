package MLP;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class Perceptron {
	double[][] inputs = new double[5][2];
	double[][] weights = new double[5][2];
	int[] labels = new int[5];
	int[] predictions = new int[5];
	double learningRate = 0.1;
	double error = 0;
	File file;
	
	public Perceptron(File f) {
		file = f;
		readFile();
		initWeights();
	}
	
	//read data from files
	public void readFile(){
		try {
			Scanner reader = new Scanner(file);
			for(int row = 0;row<5;row++) {
				for(int column = 0;column < 3;column++) {
					if(column < 2) {
						inputs[row][column] = reader.nextInt();
					}
					else {
						labels[row] = reader.nextInt();
					}
				}
			}
			reader.close();
		}
		catch(FileNotFoundException e) {
			System.out.println("An Error Occured");
			e.printStackTrace();
		}	
	}
	
	public void initWeights() {
		for(int index = 0; index<inputs.length;index++) {
			for(int i=0;i<2;i++) {
				weights[index][i] = Math.random()*2-1;
			}
		}
	}
	
	public int activation(double value) {
		if(value > 0) {
			return 1;
		}
		else {
			return -1;
		}
	}
	
	public void train() {
		double sum;
		int guess;
		int totalError;
		do{
			totalError = 0;
			for(int row = 0;row<5;row++) {
				sum = 0;
				error = 0;
				for(int index=0;index<2;index++) {
					sum = sum + inputs[row][index]*weights[row][index];
				}
				guess = activation(sum);
				error = labels[row] - guess;
				totalError += Math.abs(error); 
				for(int index=0;index<2;index++) {
					weights[row][index] = weights[row][index]+(error*inputs[row][index]*learningRate);
				}
			}
		}while(totalError>0.5);
	}
	
	public void test() {
		double sum;
		int counter = 0;
		while(counter < 5) {
			sum = 0;
			for(int index=0;index<2;index++) {
				sum = sum + inputs[counter][index]*weights[counter][index];
			}
			predictions[counter] = activation(sum);
			counter++;
		}
	}
	
	public void accuracy() {
		int correctPredictions = 0;
		double accuracy;
		for(int index = 0;index<inputs.length;index++) {
			if(predictions[index]==labels[index]) {
				correctPredictions++;
			}
		}
		accuracy = (double)correctPredictions/inputs.length;
		System.out.println("Accuracy: "+accuracy*100+" %");
	}
	
	public void printInputs() {
		for(int row = 0;row<5;row++) {
			for(int column = 0;column < 2;column++) {
				System.out.print(inputs[row][column]+", ");
			}
			System.out.println();
		}
	}
	
	public void printWeights() {
		for(int row = 0;row<5;row++) {
			for(int column = 0;column < 2;column++) {
				System.out.print(weights[row][column]+", ");
			}
			System.out.println();
		}
	}
	
	public void printLabels() {
		for(int row = 0;row<5;row++) {
			System.out.print(labels[row]+", ");
		}
	}
	public void printPredictions() {
		for(int row = 0;row<5;row++) {
			System.out.print(predictions[row]+", ");
		}
	}
}

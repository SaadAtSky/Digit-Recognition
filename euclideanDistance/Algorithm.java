package euclideanDistance;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.lang.Math;

public class Algorithm {
	//create variables for file and array to store the data values
	File fileDataset1;
	File fileDataset2;
	int[] predictions1 = new int[2810];
	int[] predictions2 = new int[2810];
	int[][] dataset1 = new int[2810][65];
	int[][] dataset2 = new int[2810][65];
	
	public Algorithm(File f1, File f2) {
		fileDataset1 = f1;
		fileDataset2 = f2;
	}
	
	//read data from files
	public void readFile(){
		try {
			Scanner reader1 = new Scanner(fileDataset1);
			Scanner reader2 = new Scanner(fileDataset2);
			for(int row = 0;row<2810;row++) {
				for(int column = 0;column < 65;column++) {
					dataset1[row][column] = reader1.nextInt();
					dataset2[row][column] = reader2.nextInt();
				}
			}
			reader1.close();
			reader2.close();
		}
		catch(FileNotFoundException e) {
			System.out.println("An Error Occured");
			e.printStackTrace();
		}	
	}
	
	//print data from array
	public void printfileDataset1() {
		for(int row = 0;row<2810;row++) {
			for(int column = 0;column < 65;column++) {
				System.out.print(dataset1[row][column]+", ");
			}
			System.out.println();
		}
	}
	public void printfileDataset2() {
		for(int row = 0;row<2810;row++) {
			for(int column = 0;column < 65;column++) {
				System.out.print(dataset2[row][column]+", ");
			}
			System.out.println();
		}
	}
	
	//calculate distance in 64 dimensions
	public double distanceCalculator(int[] point1, int[] point2) {
		double distance = 0;
		for(int index = 0;index<65;index++) {
			distance = distance + (point1[index]-point2[index])*(point1[index]-point2[index]);
		}
		return Math.sqrt(distance);
	}

	//test = Dataset1 train = Dataset2
	public void test1() {
		int counter = 0;
		int[] testSetValue = new int[65];
		int[] trainSetValue = new int[65];
		double closestValue, currentValue;
		while(counter<2810) {//iterate through rows in dataset 1
			for(int index = 0;index<65;index++) {
				testSetValue[index] = dataset1[counter][index];
			}
			for(int index = 0;index<65;index++) {
				trainSetValue[index] = dataset2[0][index];
			}
			closestValue = distanceCalculator(testSetValue,trainSetValue);
			predictions1[counter] = dataset2[0][64];//assign the category
			for(int row = 1;row<2810;row++) {//iterate through rows in dataset 2 to find the closest value to the test value
				for(int index = 0;index<65;index++) {
					trainSetValue[index] = dataset2[row][index];
				}
				currentValue = distanceCalculator(testSetValue,trainSetValue);
				if(currentValue < closestValue) {
					closestValue = currentValue;
					predictions1[counter] = dataset2[row][64];//assign the category
				}
			}
			counter++;
		}
	}
	
	//test = Dataset2 train = Dataset1
	public void test2() {
		int counter = 0;
		int[] testSetValue = new int[65];
		int[] trainSetValue = new int[65];
		double closestValue, currentValue;
		while(counter<2810) {//iterate through rows in dataset 1
			for(int index = 0;index<65;index++) {
				testSetValue[index] = dataset2[counter][index];
			}
			for(int index = 0;index<65;index++) {
				trainSetValue[index] = dataset1[0][index];
			}
			closestValue = distanceCalculator(testSetValue,trainSetValue);
			predictions2[counter] = dataset1[0][64];//assign the category
			for(int row = 1;row<2810;row++) {//iterate through rows in dataset 2 to find the closest value to the test value
				for(int index = 0;index<65;index++) {
					trainSetValue[index] = dataset1[row][index];
				}
				currentValue = distanceCalculator(testSetValue,trainSetValue);
				if(currentValue < closestValue) {
					closestValue = currentValue;
					predictions2[counter] = dataset1[row][64];//assign the category
				}
			}
			counter++;
		}
	}
	
	public void displayPredictions1() {
		for(int index = 0;index<2810;index++) {
			System.out.println(predictions1[index]);
		}
	}
	
	public void displayPredictions2() {
		for(int index = 0;index<2810;index++) {
			System.out.println(predictions2[index]);
		}
	}
	
	public void resultsAccuracy() {
		int correctPredictions=0;
		double accuracy = 0;
		for(int index = 0;index<2810;index++) {
			if(predictions1[index]==dataset1[index][64]){
				correctPredictions++;
			}
			if(predictions2[index]==dataset2[index][64]){
				correctPredictions++;
			}
		}
		accuracy = ((double)correctPredictions/5620)*100;
		System.out.println(correctPredictions+" out of "+2810*2);
		System.out.println("Accuracy: "+accuracy+" %");
	}
}

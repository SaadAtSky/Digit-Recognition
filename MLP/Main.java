package MLP;

import java.io.File;

public class Main {	
	public static void main(String[] args) {
		File file1 = new File("dataset1");
		File file2 = new File("dataset2");
		File fileDemo = new File("demo.txt");
		File fileDemo2 = new File("demo2");
		
		double[][] dataset1 = StaticFunctions.readFile1(file1);
		double[][] dataset2 = StaticFunctions.readFile2(file2);
		double[][] demo = StaticFunctions.readFile3(fileDemo);
		double[][] demo2 = StaticFunctions.readFile4(fileDemo2);
		
		MultiLayerPerceptron mlp = new MultiLayerPerceptron(64,32,10);
		double[][] inputArray;
		double[][] targetArray;

		System.out.println("Started Training");
//			mlp.learningRate = mlp.learningRate + 0.01;
		for(int index = 0;index<10000;index++) {
			for(int i =0;i<2810;i++) {
				inputArray = StaticFunctions.singleInput(dataset1,i);
				targetArray = StaticFunctions.makeTargetArray(StaticFunctions.labels1[i]);
				mlp.train(inputArray, targetArray);
			}
			//CHANGING WEIGHTS
			mlp.change();
		}
		System.out.println("Finished Training");
		
		int answerIndex=0;
		int correctPredictions=0;
		System.out.println("FINAL RESULT");
		while(answerIndex<2810) {
			double[][] output = mlp.feedforward(StaticFunctions.singleInput(dataset2,answerIndex));
			double highestValue=output[0][0];
			int prediction=0;
			for(int row=0;row<output.length;row++) {
				if(output[row][0]>highestValue) {
					highestValue = output[row][0];
					prediction = row;
				}
//				System.out.println("label "+row+": "+output[row][0]);
			}
			if(prediction==StaticFunctions.labels2[answerIndex]) {
				correctPredictions++;
			}
			System.out.println("PREDICTION: "+prediction);
			System.out.println("ANSWER: "+StaticFunctions.labels2[answerIndex]);
			answerIndex++;
		}
		System.out.println("Accuracy: "+(double)correctPredictions/2810*100);


	}

}

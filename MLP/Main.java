package MLP;

import java.io.File;

public class Main {	
	public static void main(String[] args) {
		// create 5 points with (x, y, label)
		File demoFile = new File("demo.txt");
		
		Perceptron instance = new Perceptron(demoFile);
		instance.train();
		instance.test();
		instance.accuracy();

	}

}

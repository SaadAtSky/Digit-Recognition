package euclideanDistance;

import java.io.File;

public class Main {

	public static void main(String[] args) {
		//import the two files contained the datasets
		File file1 = new File("dataset1");
		File file2 = new File("dataset2");
		
		Algorithm instance = new Algorithm(file1,file2);
		instance.readFile();
		instance.test1();
		instance.test2();
		instance.resultsAccuracy();
	}

}

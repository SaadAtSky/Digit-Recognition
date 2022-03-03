package MLP;

/*
 * Implementation of a multiclass classifier based numberOfOutputNodes a Multilayer Perceptron
 * to attempt to accurately classify hand-written digits using the MNIST Data set
 */
public class Classifier {
	private int numOfHiddenLayers;// note: all hidden layers will have the same number of nodes
	private double learningRate;
	private Matrix inputNodes, outputNodes, averageError;
	private Matrix[] hiddenNodes,hiddenWeights,hiddenBias,hiddenBiasDelta,hiddenWeightsDelta,hiddenGradient;
	private Matrix outputWeights, outputBias, outputBiasDelta, outputWeightsDelta, outputGradient;

	public Classifier(int numberOfInputNodes, int numberOfHiddenNodes, int numberOfOutputNodes, int numberOfHiddenLayers) {
		// initialize nodes, weights, biases, and gradients
		setNumberOfHiddenLayers(numberOfHiddenLayers);
		hiddenNodes = new Matrix[numOfHiddenLayers];
		hiddenWeights = new Matrix[numOfHiddenLayers];
		hiddenBias = new Matrix[numOfHiddenLayers];
		hiddenBiasDelta = new Matrix[numOfHiddenLayers];
		hiddenWeightsDelta = new Matrix[numOfHiddenLayers];
		hiddenGradient = new Matrix[numOfHiddenLayers];
		int singleColumn = 1;
		inputNodes = new Matrix(numberOfInputNodes, singleColumn);
		for (int hiddenLayerIndex = 0; hiddenLayerIndex < numOfHiddenLayers; hiddenLayerIndex++) {
			hiddenNodes[hiddenLayerIndex] = new Matrix(numberOfHiddenNodes, singleColumn);
			hiddenGradient[hiddenLayerIndex] = new Matrix(numberOfHiddenNodes, singleColumn);
			if (hiddenLayerIndex > 0) {
				hiddenWeightsDelta[hiddenLayerIndex] = new Matrix(numberOfHiddenNodes, numberOfHiddenNodes);
				hiddenWeights[hiddenLayerIndex] = new Matrix(numberOfHiddenNodes, numberOfHiddenNodes);
			} else {
				hiddenWeightsDelta[hiddenLayerIndex] = new Matrix(numberOfHiddenNodes, numberOfInputNodes);
				hiddenWeights[hiddenLayerIndex] = new Matrix(numberOfHiddenNodes, numberOfInputNodes);
			}
			randomize(hiddenWeights[hiddenLayerIndex]);
			hiddenBias[hiddenLayerIndex] = new Matrix(numberOfHiddenNodes, singleColumn);
			randomize(hiddenBias[hiddenLayerIndex]);
		}

		outputNodes = new Matrix(numberOfOutputNodes, singleColumn);
		averageError = new Matrix(numberOfOutputNodes, singleColumn);
		outputGradient = new Matrix(numberOfOutputNodes, singleColumn);
		outputWeightsDelta = new Matrix(numberOfOutputNodes, numberOfHiddenNodes);

		// initialize the weights for hidden output layer
		outputWeights = new Matrix(numberOfOutputNodes, numberOfHiddenNodes);
		randomize(outputWeights);

		// initialize bias for output layer
		outputBias = new Matrix(numberOfOutputNodes, singleColumn);
		randomize(outputBias);
	}

	// initialize the values of a matrix between 1 and -1
	public void randomize(Matrix matrix) {
		for (int row = 0; row < matrix.getRow(); row++) {
			for (int column = 0; column < matrix.getColumn(); column++) {
				matrix.getData()[row][column] = (double) Math.random() * 2 - 1;// random values between 1 and -1
			}
		}
	}

	// feed an input value to the classifier
	// to generate a predicted value
	public double[][] feedforward(double[][] inputs) {
		double[][] result;

		// initialize inputNodes
		inputNodes = Matrix.fromArray(inputs);

		// feed forward through the multiple input layers
		for (int hiddenLayerIndex = 0; hiddenLayerIndex < numOfHiddenLayers; hiddenLayerIndex++) {
			// dotProduct inputs/previous hidden layer output with hidden weights to get hidden layer
			if(hiddenLayerIndex == 0) {
				hiddenNodes[hiddenLayerIndex] = Matrix.dotProduct(hiddenWeights[hiddenLayerIndex], inputNodes);
			}
			else {
				hiddenNodes[hiddenLayerIndex] = Matrix.dotProduct(hiddenWeights[hiddenLayerIndex], hiddenNodes[hiddenLayerIndex-1] );
			}
			// add bias to result
			hiddenNodes[hiddenLayerIndex] = Matrix.add(hiddenNodes[hiddenLayerIndex], hiddenBias[hiddenLayerIndex]);

			// apply the Sigmoid activation function
			hiddenNodes[hiddenLayerIndex] = activation(hiddenNodes[hiddenLayerIndex]);
		}
		// dotProduct hidden layer with output weights to get output layer
		outputNodes = Matrix.dotProduct(outputWeights, hiddenNodes[numOfHiddenLayers - 1]);

		// add bias to result
		outputNodes = Matrix.add(outputNodes, outputBias);

		// apply the Sigmoid activation function
		outputNodes = activation(outputNodes);

		// store result value
		result = Matrix.toArray(outputNodes);

		return result;
	}

	// train the classifier on the training data by calculating the total value of
	// deltas
	// before finding the avergae delta value and chaning weights and biases
	// accordingly in the next function
	public void train(double[][] inputs, double[][] targetArray) {
		// get predicted results for the given input
		double[][] resultArray = feedforward(inputs);

		// convert results and target arrays to matrix
		Matrix result = Matrix.fromArray(resultArray);
		Matrix target = Matrix.fromArray(targetArray);

		// calculate output error = output - target and update the overall average error
		Matrix outputError = Matrix.subtract(target, result);
		averageError = Matrix.add(averageError, outputError);

		// transpose hidden weights
		Matrix transposedOutputWeights = Matrix.transpose(outputWeights);

		// calculate hidden error = transposed output weights . output error
		Matrix hiddenError = Matrix.dotProduct(transposedOutputWeights, outputError);

		// hidden gradient
		outputGradient = sigmoidDerivative(result);

		// dotProduct error
		outputGradient = Matrix.elementWise(outputError, outputGradient);

		// multiply Learning Rate
		outputGradient = Matrix.scalarMultiply(outputGradient, learningRate);

		// dotProduct transposed weights
		outputWeightsDelta = Matrix.add(outputWeightsDelta,
				Matrix.dotProduct(outputGradient, Matrix.transpose(hiddenNodes[numOfHiddenLayers - 1])));

		// adjust output bias
		outputBiasDelta = Matrix.add(outputBias, Matrix.add(outputBias, outputGradient));

		for (int hiddenLayerIndex = numOfHiddenLayers - 1; hiddenLayerIndex > -1; hiddenLayerIndex--) {
			// gradient input
			hiddenGradient[hiddenLayerIndex] = sigmoidDerivative(hiddenNodes[hiddenLayerIndex]);

			// multiple error
			hiddenGradient[hiddenLayerIndex] = Matrix.elementWise(hiddenError, hiddenGradient[hiddenLayerIndex]);

			// dotProduct learning rate
			hiddenGradient[hiddenLayerIndex] = Matrix.scalarMultiply(hiddenGradient[hiddenLayerIndex], learningRate);

			// adjust input bias
			hiddenBiasDelta[hiddenLayerIndex] = Matrix.add(hiddenBias[hiddenLayerIndex],
					Matrix.add(hiddenBias[hiddenLayerIndex], hiddenGradient[hiddenLayerIndex]));

			// dotProduct transposed weights
			if (hiddenLayerIndex == 0) {
				hiddenWeightsDelta[hiddenLayerIndex] = Matrix.add(hiddenWeightsDelta[hiddenLayerIndex],
						Matrix.dotProduct(hiddenGradient[hiddenLayerIndex], Matrix.transpose(inputNodes)));
			} else {
				hiddenWeightsDelta[hiddenLayerIndex] = Matrix.add(hiddenWeightsDelta[hiddenLayerIndex],
						Matrix.dotProduct(hiddenGradient[hiddenLayerIndex], Matrix.transpose(hiddenNodes[hiddenLayerIndex - 1])));

				// calculate hidden error for next iteration
				hiddenError = Matrix.dotProduct(Matrix.transpose(hiddenWeights[hiddenLayerIndex]), hiddenError);
			}

		}
	}

	// find the average deltas and changed weights and biases accordingly
	public void change() {
		// adjust output weights
		outputWeightsDelta = Matrix.scalarMultiply(outputWeightsDelta, (double) 1 / 2810);
		outputWeights = Matrix.add(outputWeights, outputWeightsDelta);

		// adjust hidden weights and bias
		for (int hiddenLayerIndex = 0; hiddenLayerIndex < numOfHiddenLayers; hiddenLayerIndex++) {
			hiddenWeightsDelta[hiddenLayerIndex] = Matrix.scalarMultiply(hiddenWeightsDelta[hiddenLayerIndex], (double) 1 / 2810);
			hiddenWeights[hiddenLayerIndex] = Matrix.add(hiddenWeights[hiddenLayerIndex], hiddenWeightsDelta[hiddenLayerIndex]);
			hiddenBias[hiddenLayerIndex] = Matrix.scalarMultiply(hiddenBiasDelta[hiddenLayerIndex], (double) 1 / 2810);

			// initialize
			hiddenGradient[hiddenLayerIndex] = new Matrix(hiddenGradient[hiddenLayerIndex].getRow(), 1);
			hiddenWeightsDelta[hiddenLayerIndex] = new Matrix(hiddenWeightsDelta[hiddenLayerIndex].getRow(),
					hiddenWeightsDelta[hiddenLayerIndex].getColumn());
		}

		// calculate average error
		averageError = Matrix.scalarMultiply(averageError, (double) 1 / 2810);

		// adjust output bias
		outputBias = Matrix.scalarMultiply(outputBiasDelta, (double) 1 / 2810);

		// initialize
		averageError = new Matrix(averageError.getRow(), 1);
		outputGradient = new Matrix(outputGradient.getRow(), 1);
		outputWeightsDelta = new Matrix(outputWeightsDelta.getRow(), outputWeightsDelta.getColumn());
	}
	
	// The sigmoid function
	public static double sigmoid(double value) {
		return 1 / (1 + (Math.exp(-value)));
	}
	
	// apply sigmoid function to the matrix
	public static Matrix activation(Matrix inputMatrix) {
		Matrix result = new Matrix(inputMatrix.getRow(), inputMatrix.getColumn());
		for (int row = 0; row < inputMatrix.getRow(); row++) {
			result.getData()[row][0] = sigmoid(inputMatrix.getData()[row][0]);
		}
		return result;
	}
	
	// apply the derivative of the sigmoid function to the matrix
	public static Matrix sigmoidDerivative(Matrix matrix) {
		Matrix result = new Matrix(matrix.getRow(), matrix.getColumn());
		for (int row = 0; row < matrix.getRow(); row++) {
			result.getData()[row][0] = matrix.getData()[row][0] * (1 - matrix.getData()[row][0]);
		}
		return result;
	}

	public void setNumberOfHiddenLayers(int number) {
		this.numOfHiddenLayers = number;
	}

	public void setLearningRate(double number) {
		this.learningRate = number;
	}

	public double getCostFunction() {
		double result = 0;
		for (int row = 0; row < averageError.getRow(); row++) {
			result = result + Math.pow(averageError.getData()[row][0], 2);
		}
		return result;
	}
}

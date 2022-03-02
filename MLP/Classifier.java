package MLP;

/*
 * Implementation of a multiclass classifier based on a Multilayer Perceptron
 * to attempt to accurately classify hand-written digits using the MNIST Data set
 */
public class Classifier {
	private int numOfHiddenLayers = 1;// note: all hidden layers will have the same number of nodes
	private double learningRate;
	private Matrix inputNodes, outputNodes, averageError;
	private Matrix[] hiddenNodes = new Matrix[numOfHiddenLayers];
	private Matrix[] hiddenWeights = new Matrix[numOfHiddenLayers];
	private Matrix[] hiddenBias = new Matrix[numOfHiddenLayers];
	private Matrix[] hiddenBiasDelta = new Matrix[numOfHiddenLayers];
	private Matrix outputWeights, outputBias, outputBiasDelta, outputWeightsDelta, outputGradient;
	private Matrix[] hiddenWeightsDelta = new Matrix[numOfHiddenLayers];
	private Matrix[] hiddenGradient = new Matrix[numOfHiddenLayers];

	public Classifier(int in, int hn, int on, int nOfHL) {
		// initialize nodes, weights, biases, and gradients
		setNumberOfHiddenLayers(nOfHL);
		int singleColumn = 1;
		inputNodes = new Matrix(in, singleColumn);
		for (int index = 0; index < numOfHiddenLayers; index++) {
			hiddenNodes[index] = new Matrix(hn, singleColumn);
			hiddenGradient[index] = new Matrix(hn, singleColumn);
			if (index > 0) {
				hiddenWeightsDelta[index] = new Matrix(hn, hn);
				hiddenWeights[index] = new Matrix(hn, hn);
			} else {
				hiddenWeightsDelta[index] = new Matrix(hn, in);
				hiddenWeights[index] = new Matrix(hn, in);
			}
			randomize(hiddenWeights[index]);
			hiddenBias[index] = new Matrix(hn, singleColumn);
			randomize(hiddenBias[index]);
		}

		outputNodes = new Matrix(on, singleColumn);
		averageError = new Matrix(on, singleColumn);
		outputGradient = new Matrix(on, singleColumn);
		outputWeightsDelta = new Matrix(on, hn);

		// initialize the weights for hidden output layer
		outputWeights = new Matrix(on, hn);
		randomize(outputWeights);

		// initialize bias for output layer
		outputBias = new Matrix(on, singleColumn);
		randomize(outputBias);
	}

	// initialize the values of a matrix between 1 and -1
	public void randomize(Matrix m) {
		for (int row = 0; row < m.getRow(); row++) {
			for (int column = 0; column < m.getColumn(); column++) {
				m.getData()[row][column] = (double) Math.random() * 2 - 1;// random values between 1 and -1
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
		for (int index = 0; index < numOfHiddenLayers; index++) {
			// dotProduct inputs with hidden weights to get hidden layer
			hiddenNodes[index] = Matrix.dotProduct(hiddenWeights[index], inputNodes);

			// add bias to result
			hiddenNodes[index] = Matrix.add(hiddenNodes[index], hiddenBias[index]);

			// apply the Sigmoid activation function
			hiddenNodes[index] = activation(hiddenNodes[index]);
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

		for (int index = numOfHiddenLayers - 1; index > -1; index--) {
			// gradient input
			hiddenGradient[index] = sigmoidDerivative(hiddenNodes[index]);

			// multiple error
			hiddenGradient[index] = Matrix.elementWise(hiddenError, hiddenGradient[index]);

			// dotProduct learning rate
			hiddenGradient[index] = Matrix.scalarMultiply(hiddenGradient[index], learningRate);

			// adjust input bias
			hiddenBiasDelta[index] = Matrix.add(hiddenBias[index],
					Matrix.add(hiddenBias[index], hiddenGradient[index]));

			// dotProduct transposed weights
			if (index == 0) {
				hiddenWeightsDelta[index] = Matrix.add(hiddenWeightsDelta[index],
						Matrix.dotProduct(hiddenGradient[index], Matrix.transpose(inputNodes)));
			} else {
				hiddenWeightsDelta[index] = Matrix.add(hiddenWeightsDelta[index],
						Matrix.dotProduct(hiddenGradient[index], Matrix.transpose(hiddenNodes[index - 1])));

				// calculate hidden error for next iteration
				hiddenError = Matrix.dotProduct(Matrix.transpose(hiddenWeights[index]), hiddenError);
			}

		}
	}

	// find the average deltas and changed weights and biases accordingly
	public void change() {
		// adjust output weights
		outputWeightsDelta = Matrix.scalarMultiply(outputWeightsDelta, (double) 1 / 2810);
		outputWeights = Matrix.add(outputWeights, outputWeightsDelta);

		// adjust hidden weights and bias
		for (int index = 0; index < numOfHiddenLayers; index++) {
			hiddenWeightsDelta[index] = Matrix.scalarMultiply(hiddenWeightsDelta[index], (double) 1 / 2810);
			hiddenWeights[index] = Matrix.add(hiddenWeights[index], hiddenWeightsDelta[index]);
			hiddenBias[index] = Matrix.scalarMultiply(hiddenBiasDelta[index], (double) 1 / 2810);

			// initialize
			hiddenGradient[index] = new Matrix(hiddenGradient[index].getRow(), 1);
			hiddenWeightsDelta[index] = new Matrix(hiddenWeightsDelta[index].getRow(),
					hiddenWeightsDelta[index].getColumn());
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
	public static Matrix sigmoidDerivative(Matrix a) {
		Matrix result = new Matrix(a.getRow(), a.getColumn());
		for (int row = 0; row < a.getRow(); row++) {
			result.getData()[row][0] = a.getData()[row][0] * (1 - a.getData()[row][0]);
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

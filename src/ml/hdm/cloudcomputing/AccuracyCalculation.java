package ml.hdm.cloudcomputing;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesSimple;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.Prism;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.HoeffdingTree;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instances;



public class AccuracyCalculation {
		public static BufferedReader readDataFile(String filename) {
			BufferedReader inputReader = null;
			try {
				inputReader = new BufferedReader(new FileReader(filename));
			} catch (FileNotFoundException ex) {
				System.err.println("File not found: " + filename);
			}	
			return inputReader;
		}

		public static Evaluation classify(Classifier model, Instances trainingSet, Instances testingSet) throws Exception {
			Evaluation evaluation = new Evaluation(trainingSet);
			model.buildClassifier(trainingSet);
			evaluation.evaluateModel(model, testingSet);
			return evaluation;
		}

		public static double calculateAccuracy(FastVector predictions) {
			double correct = 0;
			for (int i = 0; i < predictions.size(); i++) {
				NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
				if (np.predicted() == np.actual()) {
					correct++;
				}
			}
			return 100 * correct / predictions.size();
		}

		public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
			Instances[][] split = new Instances[2][numberOfFolds];
			for (int i = 0; i < numberOfFolds; i++) {
				split[0][i] = data.trainCV(numberOfFolds, i);
				split[1][i] = data.testCV(numberOfFolds, i);
			}
			return split;
		}

		public static void main(String[] args) throws Exception {
			BufferedReader datafile = readDataFile("documents.arff");
			Instances data = new Instances(datafile);
			data = NominalConverter.stringToNominal(data, "1-10");
			
			//Unterteilen in Training und Test Array
			data.setClassIndex(8);
			Instances[][] split = crossValidationSplit(data, 10);
			Instances[] trainingSplits = split[0];
			Instances[] testingSplits = split[1];
			
			// Angabe der Classifier
			Classifier[] models = { new J48(), new PART(), new DecisionTable(), new DecisionStump(), new Prism(), new Id3(),
					new NaiveBayesSimple(), new JRip(), new HoeffdingTree(),
					new PART()};
			// Für jedes Model ausführen
			for (int j = 0; j < models.length; j++) {
				FastVector predictions = new FastVector();
				for (int i = 0; i < trainingSplits.length; i++) {
					Evaluation validation = classify(models[j], trainingSplits[i], testingSplits[i]);
					predictions.appendElements(validation.predictions());
					
				}
				//Accurancy berechnen
				double accuracy = calculateAccuracy(predictions);
				//Ausgabe
				System.out.println("Accuracy of " + models[j].getClass().getSimpleName() + ": "
						+ String.format("%.2f%%", accuracy) + "\n---------------------------------");
			}
		}
	}



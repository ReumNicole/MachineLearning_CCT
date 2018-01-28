package ml.hdm.cloudcomputing;

import java.io.File;

import weka.associations.Apriori;
import weka.classifiers.rules.Prism;
import weka.classifiers.trees.Id3;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;

public class DocAssignment {

	public static void main(String[] args) throws Exception {

		CSVLoader loader = new CSVLoader();
		loader.setSource(new File("KatZuweisung.csv"));
		Instances data = loader.getDataSet();

		// Generierte ARFF Datei speichern
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new File("KatZuweisung.arff"));
		saver.writeBatch();


		String dataset = "KatZuweisung.arff";
		DataSource source = new DataSource(dataset);
		Instances data1 = source.getDataSet();
		//Zu klassifizierendes Objekt über den Spaltenindex setzen
		data1.setClassIndex(7);
		
		//Prism Model bauen
		Prism model = new Prism();
		model.buildClassifier(data1);
		
		Apriori model1 = new Apriori();
		model1.buildAssociations(data1);
		
		Id3 model2 = new Id3();
		model2.buildClassifier(data1);
		
		
		System.out.print(model);
		System.out.print("--------------------------");
	    System.out.print(model1);
		System.out.print("--------------------------");
    	System.out.print(model2);
		System.out.print(data);

	}
}

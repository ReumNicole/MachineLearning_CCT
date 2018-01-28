package ml.hdm.cloudcomputing;

import java.io.File;

//import required classes
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;

/*
 * Umsetzung der ersten Fallstudie, zur Erkenunng von Mustern in einem Gespräch 
 * sowie der Ableitung für an das Dokument gestellte Eigenschaften mittels dem k-Means Clustering Algorithmus.
 */

public class Clustering {
	public static void main(String args[]) throws Exception {

		// Methode zum Laden einer vorhandenen CSV Datei zwecks der
		// anschließenden Konvertierung zum geforderten ARFF Dateityp
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File("Geasprech.csv"));
		Instances data = loader.getDataSet();

		// Generierte ARFF Datei speichern
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new File("Geasprech.arff"));
		saver.writeBatch();

		// Datenset laden
		String dataset = "Geasprech.arff";
		DataSource source = new DataSource(dataset);
		Instances data1 = source.getDataSet();
		// Neue Instanz eines k-Means Cluster
		SimpleKMeans model = new SimpleKMeans();

		// Menge an k Clustern setzen
		model.setNumClusters(8);
		model.setMaxIterations(10);
		// Distanzfunktion: Per default wird eine euklidische Distanberechnung
		// durchgeführt andernfalls: model.setDistanceFunction(new weka.core.ManhattanDistance());

		// Cluster Model bauen
		model.buildClusterer(data1);
		System.out.println(model);
		
		/*
		 * Cluster Evaluation mit Test und Trainingsdaten
		 */
		ClusterEvaluation clsEval = new ClusterEvaluation();
		// Datenset laden
		String dataset1 = "Geasprech_Test.arff";
		DataSource source1 = new DataSource(dataset1);
		Instances data2 = source1.getDataSet();

		clsEval.setClusterer(model);
		clsEval.evaluateClusterer(data2);

		System.out.println("# of clusters: " + clsEval.getNumClusters());
		

	}

}

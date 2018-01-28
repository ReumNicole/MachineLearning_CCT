package ml.hdm.cloudcomputing;
import weka.classifiers.rules.Prism;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;

public class NominalConverter {

		public static void main(String[] args) throws Exception {


			DataSource source = new DataSource("test.arff");
			Instances data = new Instances(source.getDataSet());

			data = numericToNominal(data, "1");
			data = stringToNominal(data, "2-4");

			// setting class attribute
			data.setClassIndex(2);

			Prism prism = new Prism();
			prism.buildClassifier(data);

			System.out.println(prism);
			System.out.println(data);
		}

		public static Instances numericToNominal(Instances dataProcessed, String variables) throws Exception {
			weka.filters.unsupervised.attribute.NumericToNominal convert = new weka.filters.unsupervised.attribute.NumericToNominal();
			String[] options = new String[2];
			options[0] = "-R";
			options[1] = variables;
			convert.setOptions(options);
			convert.setInputFormat(dataProcessed);
			Instances filterData = Filter.useFilter(dataProcessed, convert);
			return filterData;
		}

		public static Instances stringToNominal(Instances dataProcessed, String variables) throws Exception {
			weka.filters.unsupervised.attribute.StringToNominal convert = new weka.filters.unsupervised.attribute.StringToNominal();
			String[] options = new String[2];
			options[0] = "-R";
			options[1] = variables;
			convert.setOptions(options);
			convert.setInputFormat(dataProcessed);
			Instances filterData = Filter.useFilter(dataProcessed, convert);
			return filterData;
		}

	}


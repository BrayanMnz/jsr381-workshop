package brayanmnz.jsr381.workshop.examples;

import brayanmnz.jsr381.workshop.util.DataSetExamples;
import deepnetts.data.DataSets;

import javax.visrec.ml.classification.MultiClassClassifier;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ri.ml.classification.MultiClassClassifierNetwork;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;

public class IrisFlowersClassificationExample {
    public static void main(String[] args) throws IOException {

        // Load iris data set
        DataSet dataSet = DataSetExamples.getIrisClassificationDataSet();

        //Splitting data into train and test data
        DataSet[] trainTest = DataSets.trainTestSplit(dataSet, 0.7);


        // Printing target classes
        System.out.println("Target Column names " +  Arrays.toString(trainTest[1].getTargetColumnsNames()));



        // Build multi class classifier using Deep Netts implementation of Feed Forward Network under the hood
        MultiClassClassifier<float[], String> irisClassifier = MultiClassClassifierNetwork.builder()
                .inputsNum(4)
                .hiddenLayers(16)
                .outputsNum(3)
                .maxEpochs(9000)
                .maxError(0.03f)
                .learningRate(0.01f)
                .trainingSet(trainTest[0])
                .build();


        //Printing all data on the test subset.
        trainTest[1].getItems().forEach(System.out::println);

        // Use classifier to predict class - returns a map with probabilities associated to possible classes
        Map<String, Float> shouldBeVersicolor = irisClassifier.classify(new float[] {0.1f, 0.2f, 0.3f, 0.4f}); // Sepal Length, Sepal Width, Petal length, Petal Width
        System.out.println(shouldBeVersicolor);


        Map<String, Float> shouldBeSetosa = irisClassifier.classify(new float[] {3.0f, 2.0f, 1.0f, 0.2f}); // Sepal Length, Sepal Width, Petal length, Petal Width
        System.out.println(shouldBeSetosa);


        Map<String, Float> shouldBeVirginica = irisClassifier.classify(new float[] {5.3f, 2.5f, 4.6f, 1.9f}); // Sepal Length, Sepal Width, Petal length, Petal Width
        System.out.println(shouldBeVirginica);



    }
}

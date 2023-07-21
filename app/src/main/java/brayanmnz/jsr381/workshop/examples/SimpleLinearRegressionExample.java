package brayanmnz.jsr381.workshop.examples;

import brayanmnz.jsr381.workshop.util.DataSetExamples;

import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.regression.SimpleLinearRegression;
import javax.visrec.ri.ml.regression.SimpleLinearRegressionNetwork;
import java.io.IOException;

/**
 * This example uses a Swedish Auto Insurance Dataset to predict the total
 * payment for all auto insurance claims (in thousands of Swedish Kronor),
 * given the total number of claims.
 *
 * This example shows how to instantiate, train, evaluate and use Linear Regression
 * using Machine Learning Layer from VisRec API.
 *
 * @author Zoran Sevarac
 */
public class SimpleLinearRegressionExample {

    public static void main(String[] args) throws IOException {
        // Create a DataSet object from the CSV file
        DataSet dataSet = DataSetExamples.getSwedishAutoInsuranceDataSet();

        // Build the model
        SimpleLinearRegression linReg = SimpleLinearRegressionNetwork.builder()
                                            .trainingSet(dataSet)
                                            .learningRate(0.1f)
                                            .maxError(0.01f)
                                            .build();

        // Display information about the trained model
        float slope = linReg.getSlope();
        float intercept = linReg.getIntercept();
        System.out.println("Trained Model y = " + slope + " * x + " + intercept);

        // Predict the outcome based on some input.
        float someInput = 0.10483871f;
        Float prediction = linReg.predict(someInput);
        System.out.println("predicted output for " + (someInput) + " is: " + (prediction));
    }

}

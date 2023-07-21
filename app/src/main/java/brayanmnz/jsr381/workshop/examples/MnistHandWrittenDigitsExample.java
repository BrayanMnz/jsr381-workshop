package brayanmnz.jsr381.workshop.examples;

import brayanmnz.jsr381.workshop.util.DataSetExamples;

import javax.imageio.ImageIO;
import javax.visrec.ml.classification.NeuralNetImageClassifier;
import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ml.model.ModelCreationException;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.Map;

public class MnistHandWrittenDigitsExample {

    public static void main(String[] args) throws IOException, ModelCreationException {

        // Download the dataset and calculate how much time it took
        long start = System.currentTimeMillis();
        DataSetExamples.ExampleDataSet dataSet = DataSetExamples.getMnistDataSet();
        System.out.printf("Took %d milliseconds to download and/or unzip the MNIST dataset%n", System.currentTimeMillis() - start);

        // Configuration to train the model
        ImageClassifier<BufferedImage> classifier = NeuralNetImageClassifier.builder()
                .inputClass(BufferedImage.class)
                .imageHeight(28)
                .imageWidth(28)
                .labelsFile(dataSet.getLabelsFile().toPath())
                .trainingFile(dataSet.getTrainingFile().toPath())
                .networkArchitecture(Paths.get("app/src/main/resources/mnist_arch.json")) //change this not in data set but from jsin file in resources folder?
                .exportModel(Paths.get("mnist.dnet"))
                .maxError(0.05f)
                .maxEpochs(3)
                .learningRate(0.01f)
                .build();

        // Testing the test image against the classifier.
        URL input = MnistHandWrittenDigitsExample.class.getClassLoader().getResource("00060.png");
        if (input == null) {
            throw new IOException("Input file not found in resources");
        }

        BufferedImage image = ImageIO.read(new File(input.getFile()));
        Map<String, Float> results = classifier.classify(image);

        // Print the outcome
        System.out.println(results);
    }
}

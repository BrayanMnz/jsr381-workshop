package brayanmnz.jsr381.workshop.examples;

import javax.imageio.ImageIO;
import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ml.classification.NeuralNetImageClassifier;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Map;
import javax.visrec.ml.model.ModelCreationException;


public class Duke {

    public static void main(String[] args) throws IOException, ModelCreationException {

        // Configure and train ML model to classify images
        ImageClassifier<BufferedImage> classifier = NeuralNetImageClassifier.builder()
                .inputClass(BufferedImage.class)
                .imageHeight(64)
                .imageWidth(64)
                .labelsFile(Paths.get("app/datasets/duke_and_nonduke/labels.txt")) // category labels
                .trainingFile(Paths.get("app/datasets/duke_and_nonduke/index.txt")) // list of images
                .networkArchitecture(Paths.get("app/src/main/resources/duke_net.json"))
                .exportModel(Paths.get("duke.dnet"))
                .maxError(0.05f)
                .maxEpochs(1000)
                .learningRate(0.01f)
                .build();

        // recognize image with a train model
        BufferedImage image = ImageIO.read(new File("app/datasets/duke_and_nonduke/duke/duke1.jpg"));
        Map<String, Float> results = classifier.classify(image);
        System.out.println(results);

        // recognize image with a train model
        BufferedImage image1 = ImageIO.read(new File("app/src/main/resources/duke.jpg"));
        Map<String, Float> results1 = classifier.classify(image1);
        System.out.println(results1);

        // recognize image with a train model
        BufferedImage image2 = ImageIO.read(new File("app/src/main/resources/duke_1.jpg"));
        Map<String, Float> results2 = classifier.classify(image2);
        System.out.println(results2);
    }
}
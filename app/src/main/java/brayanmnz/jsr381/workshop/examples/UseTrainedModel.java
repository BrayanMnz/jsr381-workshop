package brayanmnz.jsr381.workshop.examples;

import deepnetts.net.ConvolutionalNetwork;
import deepnetts.util.FileIO;

import javax.imageio.ImageIO;
import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ri.ml.classification.ImageClassifierNetwork;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

public class UseTrainedModel {
    public static void main(String[] args) throws IOException, ClassNotFoundException {

        // load a trained model/neural network
        ConvolutionalNetwork convNet = FileIO.createFromFile("../hotdog.dnet", ConvolutionalNetwork.class);
        // create an image classifier using trained model
        ImageClassifier<BufferedImage> classifier = new ImageClassifierNetwork(convNet);

        // load image to classify
        //BufferedImage image = ImageIO.read(new File("./src/main/resources/hot_dog.jpg"));

        BufferedImage image = ImageIO.read(new File("./src/main/resources/pizza.jpg"));
        // feed image into a classifier to recognize it
        Map<String, Float> results = classifier.classify(image);

        System.out.println(results + "\n");

        // interpret the classification result / class probability
        float hotDogProbability = results.get("hot_dog");
        if (hotDogProbability > 0.5) {
            System.out.println("There is a high probability that this is a hot dog");
        } else {
            System.out.println("Most likely this is not a hot dog");
        }

    }

}

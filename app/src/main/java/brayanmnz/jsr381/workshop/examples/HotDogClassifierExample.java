package brayanmnz.jsr381.workshop.examples;

import javax.visrec.ml.classification.NeuralNetImageClassifier;
import javax.visrec.ml.model.ModelCreationException;
import java.awt.image.BufferedImage;
import java.nio.file.Paths;

public class HotDogClassifierExample {

    public static void main(String[] args) {
        try {
            NeuralNetImageClassifier.builder()
                    .inputClass(BufferedImage.class) // input class for classifier
                    .imageWidth(128) // width of the input image
                    .imageHeight(128) // height of the input image
                    .labelsFile(Paths.get("images/labels.txt"))// list of image labels
                    .trainingFile(Paths.get("images/index.txt")) // index of images with corresponding labels
                    .networkArchitecture(Paths.get("src/main/resources/hot_dog.json"))// architecture of the convolutional neural network in json
                    .maxError(0.03f) // error level to stop the training (maximum acceptable error)
                    .maxEpochs(1000) // maximum number of training iterations (epochs)
                    .learningRate(0.01f)// amount of error to use for adjusting internal parameters in each training iteration
                    .exportModel(Paths.get("hotdog_1.dnet")) // name of the file to save trained model
                    .build();

        } catch (ModelCreationException e) { // if something goes wrong an exception is thrown
            System.out.println("Model creation failed! " + e.getMessage());
        }
    }
}

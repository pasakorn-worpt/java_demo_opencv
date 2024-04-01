package com.example.opencv;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.Feature2D;
import org.opencv.features2d.SIFT;
import org.opencv.features2d.AgastFeatureDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import javafx.scene.image.Image;
import java.io.ByteArrayInputStream;
import java.util.LinkedList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        System.loadLibrary("opencv_java454d");
        // detectFace("/home/pasakorn/Workspace/java/demo/src/main/resources/IMG_0001.jpg",
        //         "/home/pasakorn/Workspace/java/demo/src/main/resources/IMG_0001_out.jpg");
        detectKrut("/home/pasakorn/Workspace/java/demo/src/main/resources/IMG_0001.jpg",
                "/home/pasakorn/Workspace/java/demo/src/main/resources/IMG_0001_x_out.jpg");
    }

    public static void detectKrut(String sourceImagePath, String targetImagePath) {
        // Load images
        Mat sourceImage = Imgcodecs.imread(sourceImagePath);
        Mat targetImage = Imgcodecs.imread("/home/pasakorn/Workspace/java/demo/src/main/resources/400px-Garuda.jpg");

        // Convert images to grayscale
        Mat graySource = new Mat();
        Mat grayTarget = new Mat();
        Imgproc.cvtColor(sourceImage, graySource, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(targetImage, grayTarget, Imgproc.COLOR_BGR2GRAY);

        // Detect keypoints and extract descriptors
        MatOfKeyPoint keyPoints1 = new MatOfKeyPoint();
        MatOfKeyPoint keyPoints2 = new MatOfKeyPoint();
        Mat descriptors1 = new Mat();
        Mat descriptors2 = new Mat();

        SIFT sift = SIFT.create(0, 3, 0.04, 10.0, 1.6);
        sift.detectAndCompute(graySource, new Mat(), keyPoints1, descriptors1);
        sift.detectAndCompute(grayTarget, new Mat(), keyPoints2, descriptors2);

        // Match descriptors
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
        List<MatOfDMatch> matches = new LinkedList<>();
        matcher.knnMatch(descriptors1, descriptors2, matches, 2);

        // Filter good matches
        LinkedList<DMatch> goodMatchesList = new LinkedList<>();
        float ratioThreshold = 0.7f;

        for (MatOfDMatch match : matches) {
            DMatch[] matchArray = match.toArray();
            if (matchArray[0].distance < ratioThreshold * matchArray[1].distance) {
                goodMatchesList.addLast(matchArray[0]);
            }
        }

        // Draw matches
        Mat outputImage = new Mat();
        MatOfDMatch goodMatches = new MatOfDMatch();
        goodMatches.fromList(goodMatchesList);
        Features2d.drawMatches(sourceImage, keyPoints1, targetImage, keyPoints2, goodMatches, outputImage);

        // Display result
        Imgcodecs.imwrite(targetImagePath, outputImage);
    }

    public static void detectFace(String sourceImagePath, String targetImagePath) {
        Mat loadedImage = Imgcodecs.imread(sourceImagePath);
        MatOfRect facesDetected = new MatOfRect();
        CascadeClassifier cascadeClassifier = new CascadeClassifier();
        int minFaceSize = Math.round(loadedImage.rows() * 0.1f);
        cascadeClassifier.load(
                "/home/pasakorn/Workspace/java/demo/src/main/resources/haarcascades/haarcascade_frontalface_alt.xml");
        cascadeClassifier.detectMultiScale(loadedImage,
                facesDetected,
                1.1,
                3,
                Objdetect.CASCADE_SCALE_IMAGE,
                new Size(minFaceSize, minFaceSize),
                new Size());
        Rect[] facesArray = facesDetected.toArray();
        for (Rect face : facesArray) {
            Imgproc.rectangle(loadedImage, face.tl(), face.br(), new Scalar(0, 0, 255), 3);
        }
        Imgcodecs.imwrite(targetImagePath, loadedImage);
    }

    public Image mat2Img(Mat mat) {
        MatOfByte bytes = new MatOfByte();
        Imgcodecs.imencode("img", mat, bytes);
        ByteArrayInputStream inputStream = new ByteArrayInputStream(bytes.toArray());
        Image img = new Image(inputStream);
        return img;
    }
}
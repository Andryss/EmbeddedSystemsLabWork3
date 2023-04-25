import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Main {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        new Main().run();
    }

    static final String winName = "Stream";

    VideoCapture capture;
    Mat image;
    Rect boxRect;
    double[] boxRectValues;
    Mat box;
    JFrame frame;
    JLabel label;
    Mat grey;
    Mat threshold;
    Mat hierarchy;
    List<MatOfPoint> contours;
    List<MatOfPoint> approxCurves;

    private void run() {
        init();
        play();
    }

    private void init() {
        capture = new VideoCapture(0);
        image = new Mat();
        read();

        Size imageSize = image.size();

        label = new JLabel();
        update();
        frame = new JFrame(winName);
        frame.setResizable(false);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.add(label);
        frame.addKeyListener(new KeyBoxRectListener());
        frame.pack();

        boxRectValues = new double[4];
        boxRectValues[2] = imageSize.width / 10;
        boxRectValues[3] = imageSize.height / 10;
        boxRectValues[0] = imageSize.width / 2 - boxRectValues[2] / 2;
        boxRectValues[1] = imageSize.height / 2 - boxRectValues[3] / 2;
        boxRect = new Rect(boxRectValues);

        grey = new Mat();
        threshold = new Mat();
        hierarchy = new Mat();
        contours = new ArrayList<>();
        approxCurves = new ArrayList<>();
    }

    private void play() {
        frame.setVisible(true);
        while (frame.isVisible()) {
            read();
            draw();
            update();
        }
    }

    private void read() {
//        image = Imgcodecs.imread("templates.jpg");
        capture.read(image);
    }

    private void draw() {
        box = new Mat(image, boxRect);

        Imgproc.cvtColor(box, grey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(grey, threshold, 127, 255, Imgproc.THRESH_BINARY);

        contours.clear();
        Imgproc.findContours(threshold, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        approxCurves.clear();
        for (MatOfPoint contour : contours) {
            MatOfPoint2f approxCurve = new MatOfPoint2f();
            MatOfPoint2f sourceCurve = new MatOfPoint2f(contour.toArray());
            Imgproc.approxPolyDP(sourceCurve, approxCurve, 0.01 * Imgproc.arcLength(sourceCurve, true), true);
            MatOfPoint destCurve = new MatOfPoint();
            destCurve.fromList(
                    Arrays.stream(approxCurve.toArray())
                            .peek(point -> point.set(new double[]{boxRect.x + point.x, boxRect.y + point.y}))
                            .collect(Collectors.toList())
            );
            approxCurves.add(destCurve);
            if (destCurve.rows() == 4) {
                Rect boundingRect = Imgproc.boundingRect(destCurve);
                double ratio = (double) boundingRect.height / boundingRect.width;
                String text = (ratio < 0.9 || ratio > 1.1) ? "Rectangle" : "Square";
                Imgproc.putText(image, text, new Point(boundingRect.x, boundingRect.y), Imgproc.FONT_HERSHEY_PLAIN, 1, new Scalar(0, 0, 255));
            } else if (approxCurve.rows() == 3) {
                Rect boundingRect = Imgproc.boundingRect(destCurve);
                Imgproc.putText(image, "Triangle", new Point(boundingRect.x, boundingRect.y), Imgproc.FONT_HERSHEY_PLAIN, 1, new Scalar(0, 0, 255));
            }
        }

        Imgproc.drawContours(image, approxCurves, -1, new Scalar(0, 255, 0));
        Imgproc.rectangle(image, boxRect, new Scalar(0, 0, 255));
    }

    private void update() {
        label.setIcon(new ImageIcon(HighGui.toBufferedImage(image)));
    }

    class KeyBoxRectListener extends KeyAdapter {
        static final int stepSize = 10;

        @Override
        public void keyPressed(KeyEvent e) {
            int keyCode = e.getKeyCode();
            if (keyCode < 37 || keyCode > 40) return;
            switch (keyCode) {
                case 37: // arrow left
                    if (e.isControlDown()) {
                        if (boxRectValues[0] > stepSize) {
                            boxRectValues[0] -= stepSize; boxRectValues[2] += stepSize;
                        }
                    } else if (e.isAltDown()) {
                        if (boxRectValues[2] > stepSize) boxRectValues[2] -= stepSize;
                    } else {
                        if (boxRectValues[0] > stepSize) boxRectValues[0] -= stepSize;
                    }
                    break;
                case 38: // arrow up
                    if (e.isControlDown()) {
                        if (boxRectValues[1] > stepSize) {
                            boxRectValues[1] -= stepSize; boxRectValues[3] += stepSize;
                        }
                    } else if (e.isAltDown()) {
                        if (boxRectValues[3] > stepSize) boxRectValues[3] -= stepSize;
                    } else {
                        if (boxRectValues[1] > stepSize) boxRectValues[1] -= stepSize;
                    }
                    break;
                case 39: // arrow right
                    if (e.isControlDown()) {
                        if (boxRectValues[0] + boxRectValues[2] < image.width() - stepSize) boxRectValues[2] += stepSize;
                    } else if (e.isAltDown()) {
                        if (boxRectValues[2] > stepSize) {
                            boxRectValues[0] += stepSize; boxRectValues[2] -= stepSize;
                        }
                    } else {
                        if (boxRectValues[0] + boxRectValues[2] < image.width() - stepSize) boxRectValues[0] += stepSize;
                    }
                    break;
                case 40: // arrow down
                    if (e.isControlDown()) {
                        if (boxRectValues[1] + boxRectValues[3] < image.height() - stepSize) boxRectValues[3] += stepSize;
                    } else if (e.isAltDown()) {
                        if (boxRectValues[3] > stepSize) {
                            boxRectValues[1] += stepSize; boxRectValues[3] -= stepSize;
                        }
                    } else {
                        if (boxRectValues[1] + boxRectValues[3] < image.height() - stepSize) boxRectValues[1] += stepSize;
                    }
                    break;
            }
            boxRect.set(boxRectValues);
        }
    }
}

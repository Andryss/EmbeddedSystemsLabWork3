import cz.adamh.utils.NativeUtils;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Main {

    // Load opencv natives
    static {
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            try {
                NativeUtils.loadLibraryFromJar("/" + System.mapLibraryName(Core.NATIVE_LIBRARY_NAME));
            } catch (IOException exception) {
                throw new RuntimeException(exception);
            }
        }

    }

    public static void main(String[] args) {
        new Main().run();
    }

    static final String winName = "Stream";

    // Useful objects
    VideoCapture capture;
    Mat image;
    Rect boxRect;
    double[] boxRectValues;
    Mat box;
    JFrame frame;
    JLabel label;

    private void run() {
        init();
        play();
    }

    // Initialize fields and main JFrame
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
        frame.addKeyListener(new KeyBoxRectListener());

        frame.add(label);
        frame.pack();

        boxRectValues = new double[4];
        boxRectValues[2] = imageSize.width / 10;
        boxRectValues[3] = imageSize.height / 10;
        boxRectValues[0] = imageSize.width / 2 - boxRectValues[2] / 2;
        boxRectValues[1] = imageSize.height / 2 - boxRectValues[3] / 2;
        boxRect = new Rect(boxRectValues);
    }

    // Main cycle
    private void play() {
        frame.setVisible(true);
        while (frame.isVisible()) {
            read();
            draw();
            update();
        }
    }

    // Read image frame and write it to "image" Mat
    private void read() {
        // image = Imgcodecs.imread("templates0.jpg");
        capture.read(image);
    }

    // Useful objects for "draw" function
    Mat grey = new Mat();
    Mat threshold = new Mat();
    Mat hierarchy = new Mat();
    List<MatOfPoint> contours = new ArrayList<>();
    List<MatOfPoint> approxCurves = new ArrayList<>();

    // Colors
    Scalar red = new Scalar(0, 0, 255);
    Scalar green = new Scalar(0, 255, 0);
    Scalar hsvLowerBound = new Scalar(0, 100, 100);
    Scalar hsvUpperBound = new Scalar(180, 255, 255);

    // Draw on "image" Mat everything we want
    private void draw() {
        // Read box mat
        box = image.submat(boxRect);

        // Convert to grey and threshold
        Imgproc.cvtColor(box, grey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(grey, threshold, 127, 255, Imgproc.THRESH_BINARY);

        // Find contours
        contours.clear();
        Imgproc.findContours(threshold, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        approxCurves.clear();
        // Iterate through contours to decide which of them are figures
        for (MatOfPoint contour : contours) {
            Point[] points = contour.toArray();
            // Don't detect if some points on the borders of the box
            if (Arrays.stream(points).anyMatch(point ->
                    point.x == 0 || point.x == boxRect.width || point.y == 0 || point.y == boxRect.height)
            ) continue;
            // Don't detect small contours
            Rect rect = Imgproc.boundingRect(contour);
            if (rect.height < 30 || rect.width < 30) continue;

            // Approximate curve with some variance
            MatOfPoint2f approxCurve = new MatOfPoint2f();
            MatOfPoint2f sourceCurve = new MatOfPoint2f(points);
            Imgproc.approxPolyDP(sourceCurve, approxCurve, 0.01 * Imgproc.arcLength(sourceCurve, true), true);
            // Add offset of the box to contour
            MatOfPoint destCurve = new MatOfPoint();
            destCurve.fromList(
                    Arrays.stream(approxCurve.toArray())
                            .peek(point -> point.set(new double[]{boxRect.x + point.x, boxRect.y + point.y}))
                            .collect(Collectors.toList())
            );
            // Append approximated curve to the list
            approxCurves.add(destCurve);

            // Get color name and figure name and put it next
            String colorAndFigure = recognizeColorAndFigure(destCurve);
            if (colorAndFigure != null) {
                Rect boundingRect = Imgproc.boundingRect(destCurve);
                Imgproc.putText(image, colorAndFigure, new Point(boundingRect.x, boundingRect.y), Imgproc.FONT_HERSHEY_PLAIN, 1, red);
            }
        }

        // Draw box rectangle
        Imgproc.rectangle(image, boxRect, red);
        // Draw approximated contours
        Imgproc.drawContours(image, approxCurves, -1, green, 2);
    }


    Mat boundingMatHsv = new Mat();
    Mat boundingMatHsvMask = new Mat();
    double[] emptyHsvScalarValue = new double[]{0, 0, 0, 0};

    // Recognize color and figure of the curve
    private String recognizeColorAndFigure(MatOfPoint destCurve) {
        // Get points count
        int rows = destCurve.rows();
        if (rows != 3 && rows != 4) return null;
        // Get curve rect
        Rect boundingRect = Imgproc.boundingRect(destCurve);

        // Convert to HSV
        Mat boundingMat = image.submat(boundingRect);
        Imgproc.cvtColor(boundingMat, boundingMatHsv, Imgproc.COLOR_BGR2HSV);
        // Get colored mask (delete white)
        Core.inRange(boundingMatHsv, hsvLowerBound, hsvUpperBound, boundingMatHsvMask);
        // Get mean color and extract hue
        Scalar mean = Core.mean(boundingMatHsv, boundingMatHsvMask);
        if (Arrays.equals(mean.val, emptyHsvScalarValue)) return null;
        double hue = mean.val[0];

        // Encode color string from hue value
        String color = "unknown";
        if (hue > 40 && hue < 75) {
            color = "green";
        } else if (hue > 90 && hue < 140) {
            color = "blue";
        } else if (hue < 10 || hue > 165) {
            color = "red";
        }

        // Encode figure string from points count
        String figure;
        if (rows == 4) {
            double ratio = (double) boundingRect.height / boundingRect.width;
            figure = (ratio < 0.9 || ratio > 1.1) ? "rectangle" : "square";
        } else {
            figure = "triangle";
        }

        // Return color and figure
        return color + " " + figure;
    }

    // Update JFrame user see
    private void update() {
        label.setIcon(new ImageIcon(HighGui.toBufferedImage(image)));
    }

    // KeyListener to move box and change box size
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

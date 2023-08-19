#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>

int main() {
    // Open webcam
    cv::VideoCapture cap(0);  // 0 corresponds to the default webcam

    if (!cap.isOpened()) {
        std::cerr << "Error opening webcam." << std::endl;
        return -1;
    }

    // Load pre-trained YOLOv3 model
    cv::dnn::Net net = cv::dnn::readNetFromDarknet("models/yolov3.cfg", "models/yolov3.weights");

    // Load class names
    std::vector<std::string> classes;
    std::ifstream classFile("models/coco.names");
    std::string className;
    while (std::getline(classFile, className)) {
        classes.push_back(className);
    }

    // Physical dimensions of the object in centimeters
    double physicalWidth = 10.0;   // Example width in centimeters
    double physicalHeight = 5.0;   // Example height in centimeters

    while (true) {
        cv::Mat frame;
        cap >> frame;  // Capture a frame from the webcam

        if (frame.empty()) {
            break;
        }

        // Prepare input blob
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(416, 416), cv::Scalar(), true, false);

        // Set input blob
        net.setInput(blob);

        // Forward pass
        std::vector<cv::Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        for (const auto& out : outs) {
            for (int i = 0; i < out.rows; ++i) {
                cv::Mat detection = out.row(i);
                cv::Point classIdPoint;
                double confidence;
                cv::minMaxLoc(detection.colRange(5, detection.cols), nullptr, &confidence, nullptr, &classIdPoint);

                if (confidence > 0.5) {
                    int classId = classIdPoint.x;
                    int centerX = static_cast<int>(detection.at<float>(0) * frame.cols);
                    int centerY = static_cast<int>(detection.at<float>(1) * frame.rows);
                    int width = static_cast<int>(detection.at<float>(2) * frame.cols);
                    int height = static_cast<int>(detection.at<float>(3) * frame.rows);

                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    // Calculate the conversion factors
                    double widthConversionFactor = physicalWidth / width;   // cm/pixel
                    double heightConversionFactor = physicalHeight / height; // cm/pixel

                    // Convert the pixel dimensions to centimeters
                    double objectWidthCm = width * widthConversionFactor;
                    double objectHeightCm = height * heightConversionFactor;

                    // Display the dimensions in centimeters
                    std::string dimensionText = "W: " + std::to_string(objectWidthCm) + " cm, H: " + std::to_string(objectHeightCm) + " cm";
                    cv::rectangle(frame, cv::Rect(left, top, width, height), cv::Scalar(0, 255, 0), 2);
                    cv::putText(frame, dimensionText, cv::Point(left, top + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
                }
            }
        }

        // Display the processed frame
        cv::imshow("Object Detection", frame);

        // Exit loop if 'q' is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release the webcam and close the window
    cap.release();
    cv::destroyAllWindows();

    return 0;
}

#include <opencv2/opencv.hpp>

int main() {
    // Load the pre-trained face detection model
    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load("/home/venkat/test/facedetection/haarcascade_frontalface_default.xml")) {
        std::cout << "Error loading face detection model." << std::endl;
        return -1;
    }

    // Load the input image
    cv::Mat image = cv::imread("2.jpg");

    // Check if the image loaded successfully
    if (image.empty()) {
        std::cout << "Error loading input image." << std::endl;
        return -1;
    }

    // Convert the image to grayscale for face detection
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // Detect faces in the grayscale image
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(grayImage, faces, 1.1, 4);

    // Draw rectangles around the detected faces
    for (const cv::Rect& faceRect : faces) {
        cv::rectangle(image, faceRect, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("facedetection.png",image);
    // Display the image with detected faces
    cv::imshow("Eye Detection", image);
    cv::waitKey(0);

    // Close the window
    cv::destroyAllWindows();

    return 0;
}

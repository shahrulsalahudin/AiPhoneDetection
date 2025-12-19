#include "net.h"
#include <opencv2/opencv.hpp>

int main()
{
    // Load NCNN model
    ncnn::Net net;
    net.load_param("/home/pi/yolo/model.param");
    net.load_model("/home/pi/yolo/model.bin");

    // Open the camera
    cv::VideoCapture cap(0); // 0 for default camera
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // Preprocess: Convert frame to NCNN format
        ncnn::Mat in = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);

        // Run inference
        ncnn::Extractor ex = net.create_extractor();
        ex.input("input", in);

        ncnn::Mat out;
        ex.extract("output", out);

        // Post-process and display results
        // (Add code to process 'out' and display bounding boxes or other info)

        cv::imshow("Detection", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

#include "net.h"
#include <opencv2/opencv.hpp>
#include <chrono>  // For measuring time

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

    // Variables for FPS calculation
    int frame_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

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

        // Update frame count and calculate FPS
        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = current_time - start_time;

        if (elapsed_time.count() >= 1.0) {  // Every 1 second
            float fps = frame_count / elapsed_time.count();
            frame_count = 0;
            start_time = current_time;

            // Display FPS on the frame
            std::string fps_text = "FPS: " + std::to_string(fps);
            cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }

        // Display the frame with FPS
        cv::imshow("Detection", frame);

        // Exit if 'q' is pressed
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

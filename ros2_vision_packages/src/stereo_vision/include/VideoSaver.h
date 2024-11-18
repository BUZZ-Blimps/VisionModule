#pragma once

#include <string>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


class VideoSaver{

    std::string filename;
    double fps;
    bool verboseMode;
    cv::VideoWriter outputVideo;

    public:
        void initVideoSaver(std::string filename, double fps, bool verboseMode){
            this->filename = filename;
            this->fps = fps;
            this->verboseMode = verboseMode;
        }

        void initVideoSaver(double fps, bool verboseMode) {
            const char* capturesDir = "/home/opi/captures";

            // Create the directory if it doesn't exist
            struct stat st = {0};
            if (stat(capturesDir, &st) == -1) {
                if (verboseMode) fprintf(stdout, "Creating directory: %s\n", capturesDir);
                if (mkdir(capturesDir, 0700) == -1) {
                    if (verboseMode) fprintf(stderr, "Error creating directory: %s\n", strerror(errno));
                }
            }

            this->filename = std::string(capturesDir) + "/video_" + getCurrentTimeEST() + ".avi";
            this->fps = fps;
            this->verboseMode = verboseMode;

            if (verboseMode) fprintf(stdout, "Video will be saved as: %s\n", this->filename.c_str());
        }

        void writeFrame(cv::Mat frame){
            if(!outputVideo.isOpened()){
                if(verboseMode) fprintf(stdout, "Trying to open video (%s).\n", filename.c_str());
                outputVideo.open(filename, cv::VideoWriter::fourcc('M','J','P','G'), fps, frame.size(), true);
            }

            if(outputVideo.isOpened()){
                if(!frame.empty()){
                    outputVideo.write(frame);
                } else{
                    if(verboseMode) fprintf(stdout, "Attempted to write empty frame.\n");
                }
            }
            
        }

        ~VideoSaver(){
            if(verboseMode) fprintf(stdout, "Closing video (%s).", filename.c_str());
            outputVideo.release();
        }
    
    private:
        std::string getCurrentTimeEST() {
            auto now = std::chrono::system_clock::now();
            auto in_time_t = std::chrono::system_clock::to_time_t(now);

            std::stringstream ss;
            ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
            return ss.str();
        }
};

/**
* This file is a modified version of ORB-SLAM2.<https://github.com/raulmur/ORB_SLAM2>
*
* This file is part of D.
* Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/bertabescos/DynaSLAM>.
*
*/

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <unistd.h>
#include <opencv2/core/core.hpp>


#include "MaskNet.h"
#include <System.h>

#include "Tracking.h"
#include "KeyFrame.h"
#include "Semantic.h"
#include "SlamConfig.h"

using namespace std;
using namespace ORB_SLAM2;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char *argv[])
{
    //FLAGS_log_dir = "/mnt/lySLAM/GLOG";
    //google::InitGoogleLogging(argv[0]);

    if(argc < 5 )
    {
        cerr << endl << "Usage: ./rgbd_tum_optical path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
        return 1;
    }

    std::cout << "===========================" << std::endl;
    std::cout << "argv[1] path_to_vocabulary: " << argv[1] << std::endl;
    std::cout << "argv[2] path_to_settings: " << argv[2] << std::endl;
    std::cout << "argv[3] path_to_sequence: " << argv[3] << std::endl;
    std::cout << "argv[4] path_to_association: " << argv[4] << std::endl;
    std::cout << "argv[5] result path: " << argv[5] << std::endl;

    // LOG(INFO) << "---------Parameters---------------";
    // LOG(INFO) << "argv[1]: " << argv[1];
    // LOG(INFO) << "argv[2]: " << argv[2];
    // LOG(INFO) << "argv[3]: " << argv[3];

    // LOG(INFO) << "argv[4]: " << argv[4];
    // LOG(INFO) << "argv[5] result path: " << argv[5];
    // LOG(INFO) << "----------------------------------";
    std::cout << "===========================" << std::endl;

    // save result
    if (argc == 6) {
        Config::GetInstance()->IsSaveResult(true);
        Config::GetInstance()->createSavePath(std::string(argv[5]));
    } else {
        Config::GetInstance()->IsSaveResult(false);
    }

    //std::thread(spin_thread);

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);

    Config::GetInstance()->LoadTUMDataset(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    std::cout << "nImages: " << nImages << std::endl;
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    std::string cnn_method = "maskrcnn"; // maskrcnn
    // SegNet case do not need set the "init_delay" and "frame_delay"
    // delay for the initial few frames
    float init_delay = 0.05; //usec
    int init_frames = 2;    //usec
    // delay for each frame
    float frame_delay = 0;

    //nh.getParam("cnn_method", cnn_method);
    // control the framerate
    //nh.getParam("init_delay", init_delay);
    //nh.getParam("frame_delay", frame_delay);
    //nh.getParam("init_frames", init_frames);
    LOG(INFO) << "Delay for the initial few frames: " << init_delay;
    //cout << "Delay for the initial few frames: " << init_delay << endl;

    Semantic::GetInstance()->SetSemanticMethod(cnn_method);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD, false);

    // 运行语义线程
    Semantic::GetInstance()->Run();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    
    cv::Mat imRGB, imD;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni], CV_LOAD_IMAGE_UNCHANGED);
        imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the image to the SLAM system
        SLAM.TrackRGBD(imRGB, imD, tframe);

        // Manually add delay to evaluate TUM, because TUM dataset is very short
        if (ni < init_frames) {
            usleep(init_delay);
        } else {
            usleep(frame_delay);
        }


        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    std::cout << "===============Tracking Finished============" << std::endl;

    std::cout << "===============Final Stage============" << std::endl;

    // Stop semantic thread
    Semantic::GetInstance()->RequestFinish();

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "------median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "------mean tracking time: " << totaltime/nImages << endl;


    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}
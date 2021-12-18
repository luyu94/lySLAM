/* 
 * Copyright (C) 2021, Yubao Liu, AISL, TOYOHASHI UNIVERSITY of TECHNOLOGY 
 * Email: yubao.liu.ra@tut.jp 
 * 
 */

#include "Semantic.h"
#include "MaskNet.h"
#include "SlamConfig.h"

#define DEBUG 0

using namespace std;
namespace ORB_SLAM2 {

void Semantic::Run()
{
    mptSemanticSegmentation = new std::thread(&Semantic::SemanticSegmentationThread, this);
    mptSemanticTracking = new std::thread(&Semantic::SemanticTrackingThread, this);
    // This thread seems not a must. I did not debug this thread in this sample code
    // This thread is described in the paper. Please try to enable it if you really need it.
    // mptSemanticBA = new std::thread(&Semantic::SemanticBAThread, this);
}

// wait semantic label for each keyframe
void Semantic::SemanticTrackingThread()
{
    KeyFrame* lastKF;
    KeyFrame* currentKF;
    KeyFrame* latestKF;
    size_t lastProcessedId = 0;
    bool bSkip = false;
    int frameCount = -1;
    int lastSegID = -1;
    LOG(INFO) << "=================Start Semantic tracking thread=============";
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    while (1) 
    {
        // new task is comming
        if (!CheckNewSemanticTrackRequest()) {  //如果语义跟踪线程没有发出请求
            if (CheckFinish()) {    //检查完毕说明语义线程停止
                std::cout << "Semantic thread stopping" << std::endl;
                LOG(INFO) << "Semantic thread Stopping ....";
                break;
            } // check finish
            else {      //没有检查完毕，就等待检查
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
        } // Is process request comming?

        // Main loop, new request came
        LOG(INFO) << "Size of semantic optimization queue: " << mlSemanticTrack.size();
        std::cout << "Size of semantic opotimization queue: " << mlSemanticTrack.size() << std::endl;

        // std::cout << "------------------Semantic tracker:" << currentKF->mnId << "-------------------------" << std::endl;
        // LOG(INFO) << "------------------Semantic tracker:" << currentKF->mnId << "-------------------------";

        // 发出跟踪请求send request, get semantic label, mLabel
        std::unique_lock<std::mutex> lock(mMutexSemanticTrack);
        currentKF = mlSemanticTrack.front();
        mlSemanticTrack.pop_front();
        if (!currentKF) {
            LOG(WARNING) << "Null key fame";
            lock.unlock();
            continue;
        }
        // currentKF->SetNotErase();
        lock.unlock();

        // 如果当前关键帧的ID大于2？？？
        if (currentKF->mnId > 2) {
            std::cout << "Optimize KF: " << currentKF->mnId << std::endl;
            // Only optimize camera pose
            auto start = std::chrono::steady_clock::now();
            mpTracker->PoseOptimization(currentKF, false);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end - start;
            mvTimeSemanticOptimization.push_back(diff.count());

            // optimize the map points at the same time
            // This maybe cause unstable of the system
            // mpTracker->PoseOptimization(currentKF, true);

            // [Debugging] Semantic BA
            // AddSemanticBARequest(currentKF);
        }

        lastKF = currentKF;
        frameCount++;
        // End while
    }
    LOG(INFO) << "==============Semantic tracking thread finished================";
}


// wait semantic label for each keyframe
void Semantic::SemanticSegmentationThread()
{
    KeyFrame *pKF;

    std::vector<cv::Mat> vLabel;    // 语义分割后的图片
    vLabel.reserve(mBatchSize);     // ???

    std::vector<cv::Mat> vRequest;   //存放关键帧对应图片
    std::vector<KeyFrame*> vKFs;    //存放关键帧指针
    vRequest.reserve(mBatchSize);
    vKFs.reserve(mBatchSize);
    size_t ID;

    LOG(INFO) << "==========Start Semantic Segmentation thread==========";

    // Initialize Mask R-CNN
    DynaSLAM::SegmentDynObject *MaskNet;
    cout << "Loading Mask R-CNN. This could take a while..." << endl;
    MaskNet = new DynaSLAM::SegmentDynObject();     //创建语义分割函数对象
    cout << "Mask R-CNN loaded!" << endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    while (1) 
    {
        // check new request
        if (!CheckNewKeyFrames()) {     //检查是否有新的关键帧，没有就继续检查
            if (CheckFinish()) {    // 检查是否已经有外部线程请求终止当前线程
                LOG(INFO) << "Semantic thread Stopping ....";
                break;
            } else {    //没有线程终止请求就不执行后续代码，继续检查是否有新的关键帧，
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            } // end check finish
        }     // end check new task comming
        LOG(INFO) << "=================================================";
        std::cout << "=================================================" << std::endl;

        //有新的关键帧到来
        vRequest.clear();
        vKFs.clear();
        vLabel.clear();
        // vScore.clear();

        std::unique_lock<std::mutex> lock(mMutexNewKFs);
        for (std::list<KeyFrame*>::iterator it = mlNewKeyFrames.begin(); it != mlNewKeyFrames.end(); it++)
        {
            pKF = *it;
            ID = pKF->mnId;
            // pKF->SetNotErase();
            if (pKF->IsSemanticReady())     //？？？如果当前帧语义分割准备好
            {
                mnTotalSemanticFrameNum++;
                AddSemanticTrackRequest(*it);
                it = mlNewKeyFrames.erase(it);
                continue;
            }

            vKFs.push_back(pKF);    //*****有新的关键帧指针就存放在vKFs中
            vRequest.push_back(pKF->mImRGB);    //*****关键帧图像放入vRequest中
        }

        // key frame queue
        lock.unlock();

        // request segmentation
        // mMaskRCNN->Segment(vRequest, vLabel, vScore, out_object_num);
        //====================进行语义分割得到分割结果============================
        MaskNet->SemanticSegmentation(vRequest, vLabel);


        // save semantic results and generate mask image
        for (size_t i = 0; i < vLabel.size(); i++) {
            vKFs[i]->mImLabel = vLabel[i];      //当前语义标签存入关键帧标签中
            this->GenerateMask(vKFs[i], true);
            // Must inform semantic ready before updating moving probability
            vKFs[i]->InformSemanticReady(true);
            vKFs[i]->UpdatePrioriMovingProbability();
            if (vKFs[i]->mnFrameId > mnLatestSemanticKeyFrameID) {
                mnLatestSemanticKeyFrameID = vKFs[i]->mnFrameId;
            }
            Config::GetInstance()->saveImage(vLabel[i], "label", std::to_string(vKFs[i]->mnId) + ".png"); //??
        }
        LOG(INFO) << "=========Size of semantic queue: " << mlNewKeyFrames.size();
        std::cout << "=========Size of semantic queue: " << mlNewKeyFrames.size() << std::endl;
    }
}

// [TODO] debugging. wait semantic label for each keyframe
void Semantic::SemanticBAThread()
{
    KeyFrame* pKF;
    LOG(INFO) << "Start Semantic BA thread";
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    while (1) {
        // new task is comming
        if (!CheckSemanticBARequest()) {
            if (CheckFinish()) {
                std::cout << "Semantic BA thread stopping" << std::endl;
                LOG(INFO) << "Semantic BA thread Stopping ....";
                break;
            } // check finish
            else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
        } // Is process request comming?

        LOG(INFO) << "Size of semantic BA queue: " << mlSemanticBA.size();
        // std::cout << "Size of semantic BA queue: " << mlSemanticBA.size() << std::endl;
        // std::cout << "-----------Semantic BA:" << pKF->mnId << "-----------------" << std::endl;
        LOG(INFO) << "-----------Semantic BA:" << pKF->mnId << "------------------";

        // send request, get semantic label, mLabel
        std::unique_lock<std::mutex> lock(mMutexSemanticBA);
        pKF = mlSemanticBA.front();
        mlSemanticBA.pop_front();
        if (!pKF) {
            LOG(WARNING) << "Null key fame";
            lock.unlock();
            continue;
        }
        lock.unlock();

        if (pKF->mnId > 3) {
            // std::cout << "Optimize KF: " << pKF->mnId << std::endl;
            mpTracker->SemanticBA(pKF);
        }
    } // End while
    LOG(INFO) << "---------Semantic tracking thread finished---------------";
}

// should be called after locking the map
// 判断是否是动态地图点
bool Semantic::IsDynamicMapPoint(const MapPoint* pMP)
{
    if (!pMP)
        return true;
    // TODO change 0.5 to a parameter (>=0.5),
    // this parameter seems also related to the semantic segmentation accuracy
    // In the paper of RDS-SLAM, 0.4 and 0.6 are used for static and dynamic threshold respectly.
    // Actually, this parameter needs to be adjusted according to your machine power and the semantic segmentation accuracy. 
    // You can set it to 0.5 if you cannot find a proper threshold.
    // Here, simplly judge dynamic objects if its probability larger than 0.5 (for example)
    // There is a similar function in Mappoint cpp, you can modify it if needed

    //============TODO===判断动态点======================
    if (pMP->mMovingProbability <= mthDynamicThreshold) {
        return false;
    }
    return true;
    // if (pMP->mMovingProbability <= 0.5) {
    //     return false;
    // }
    // return true;
}

size_t Semantic::GetLatestSemanticKeyFrame()
{
    return mnLatestSemanticKeyFrameID;
}

//插入关键帧
void Semantic::InsertKeyFrame(KeyFrame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
}

//插入语义关键帧请求
void Semantic::InsertSemanticRequest(KeyFrame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutexNewSemanticRequest);
    if (!pKF) {
        LOG(ERROR) << "pKF is null";
    }
    pKF->mbIsInsemanticQueue = true;
    // if (pKF->isBad()) {
    //     LOG(ERROR) << "pKF is bad";
    // }
    LOG(INFO) << "Insert semantic request: " << pKF->mnId;
    // pKF->SetNotErase();
    mlNewSemanticRequest.push_back(pKF);
}

void Semantic::IsEnableSemantic(bool in_isEnable)
{
    mbIsUseSemantic = in_isEnable;
}

//加入语义跟踪请求
void Semantic::AddSemanticTrackRequest(KeyFrame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutexSemanticTrack);
    mlSemanticTrack.push_back(pKF);
}

//加入语义BA请求
void Semantic::AddSemanticBARequest(KeyFrame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutexSemanticBA);
    mlSemanticBA.push_back(pKF);
}

bool Semantic::CheckSemanticBARequest()
{
    std::unique_lock<std::mutex> lock(mMutexSemanticBA);
    return !(mlSemanticBA.empty()) && mlSemanticBA.size() > 5;
}

//检查新的语义跟踪请求
bool Semantic::CheckNewSemanticTrackRequest()
{
    std::unique_lock<std::mutex> lock(mMutexSemanticTrack);
    return !(mlSemanticTrack.empty());
}

//检查新的关键帧
bool Semantic::CheckNewKeyFrames()
{
    if (mbIsUseSemantic == false) {
        return false;
    }
    std::unique_lock<std::mutex> lock(mMutexNewKFs);
    bool ret = (mlNewKeyFrames.size() >= mBatchSize) ? true : false;        //？？？？
    return ret;
}

//检查新的语义请求
bool Semantic::CheckNewSemanticRequest()
{
    std::unique_lock<std::mutex> lock(mMutexNewSemanticRequest);
    bool ret = (mlNewSemanticRequest.size() >= mBatchSize) ? true : false;  //???
    return ret;
}

//===========???
void Semantic::RequestFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinishRequested = true;
    LOG(INFO) << "Semantic thread request stop";
    lock.unlock();

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    if (mptSemanticSegmentation->joinable())    //判断线程是否可执行
        mptSemanticSegmentation->join();        //阻塞当前线程

    if (mptSemanticTracking->joinable())
        mptSemanticTracking->join();

    // for debugging
    // FinalStage();
}

// void Semantic::FinalStage()
// {
//     LOG(INFO) << "[Semantic] FinalStage, some results are saved at ~/.ros";
//     // To evaluate the time delay between sequential model and bi-direction model
//     std::ofstream fSemanticDelay;
//     fSemanticDelay.open("SemanticDelay.txt");
//     fSemanticDelay << "FrameID"
//                    << ","
//                    << "Delay\n";
//     int nTotalSemanticDelay = -1;
//     for (size_t i = 0; i < mvSemanticDelay.size(); i++) {
//         fSemanticDelay << i << "," << mvSemanticDelay[i] << "\n";
//         nTotalSemanticDelay += mvSemanticDelay[i];
//     }
//     LOG(INFO) << "Average Semantic Delay: " << nTotalSemanticDelay / mvSemanticDelay.size() << " Frames";
//     std::cout << "Average Semantic Delay: " << nTotalSemanticDelay / mvSemanticDelay.size() << " Frames" << std::endl;
//
//     // ----------------------------------------
//     // Total keyframe number that segmented
//     // Please shutdown SLAM once tracking is finished, otherwise this value will be not accurate
//     LOG(INFO) << "Total semantic KeyFrame Num: " << mnTotalSemanticFrameNum;
//     std::cout << "Total semantic KeyFrame Num: " << mnTotalSemanticFrameNum << std::endl;
//
//     // Time evaluation of mvoving probability updating model
//     float nToalTimeUpdateMovingProbability = 0;
//     for (size_t i = 0; i < mvTimeUpdateMovingProbability.size(); i++) {
//         nToalTimeUpdateMovingProbability += (float)mvTimeUpdateMovingProbability[i];
//     }
//     LOG(INFO) << "Average time of moving probability updating: " << nToalTimeUpdateMovingProbability / mvTimeUpdateMovingProbability.size() * 1000 << " ms";
//     std::cout << "Average time of moving probability updating: " << nToalTimeUpdateMovingProbability / mvTimeUpdateMovingProbability.size() * 1000 << " ms" << std::endl;
//
//     // Time evaluation of mask generation
//     float nTotalTimeMaskGeneration = 0;
//     for (size_t i = 0; i < mvTimeMaskGeneration.size(); i++) {
//         nTotalTimeMaskGeneration += (float)mvTimeMaskGeneration[i];
//     }
//     LOG(INFO) << "Average time of mask generation: " << nTotalTimeMaskGeneration / mvTimeMaskGeneration.size() * 1000 << " ms";
//     std::cout << "Average time of mask generation: " << nTotalTimeMaskGeneration / mvTimeMaskGeneration.size() * 1000 << " ms" << std::endl;
//
//     // TIme evaluation for semanti-based Optimization
//     float nTotalSemanticBasedOptimization = 0;
//     for (int i = 0; i < mvTimeSemanticOptimization.size(); i++) {
//         nTotalSemanticBasedOptimization += mvTimeSemanticOptimization[i];
//     }
//     LOG(INFO) << "Average time of semantic optimization: " << nTotalSemanticBasedOptimization / mvTimeMaskGeneration.size() * 1000 << " ms";
//     std::cout << "Average time of semantic optimization: " << nTotalSemanticBasedOptimization / mvTimeMaskGeneration.size() * 1000 << " ms" << std::endl;
// }

// 检查是否已经有外部线程请求终止当前线程
bool Semantic::CheckFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

//立即得到语义信息，创建Semantic()对象
Semantic* Semantic::GetInstance()
{
    if (mInstance == nullptr) {
        // mInstance = std::make_shared<Semantic>();
        mInstance = new Semantic();
    }
    return mInstance;
}

void Semantic::SetSemanticMethod(const std::string& cnn_method)
{
    msCnnMethod = cnn_method;
    LOG(INFO) << "Use CNN method: " << msCnnMethod;
    std::cout << "Use CNN method: " << msCnnMethod << std::endl;
    if (msCnnMethod == "maskrcnn") {
        mmDynamicObjects.insert({ "PEOPLE", 1 });
    } else {
        if (msCnnMethod == "segnet") {
            mmDynamicObjects.insert({ "PEOPLE", 15 });
        }
    }
}


//各参数
Semantic::Semantic()
{
    //msCnnMethod = "segnet"; // or "maskrcnn"
    mnLatestSemanticKeyFrameID = 0;
    mnTotalSemanticFrameNum = 0;

    // statistics
    // Resrve space of total frame number int TUM dataset
    mvTimeUpdateMovingProbability.reserve(1000);
    mvTimeMaskGeneration.reserve(1000);
    mvTimeSemanticOptimization.reserve(1000);

    mBatchSize = 2;
    mbFinishRequested = false;

    // Morphological filter
    mDilation_size = 15;
    mKernel = getStructuringElement(cv::MORPH_ELLIPSE,
        cv::Size(2 * mDilation_size + 1, 2 * mDilation_size + 1),
        cv::Point(mDilation_size, mDilation_size));

    // threshold for moving probablility of map points
    mthDynamicThreshold = 0.5;
    // segment the first few frames
    mnSegmentFrameNum = 1;
    mbIsUseSemantic = true;
    // Start a semantic tracing thread
    LOG(INFO) << "Create Semantic instance and start new thread";
}

//产生Mask
void Semantic::GenerateMask(KeyFrame* pKF, const bool isDilate)
{
    if (pKF->mImLabel.empty() || pKF->mImLabel.data == nullptr) {
        // pKF->InformSemanticReady(false);
        LOG(WARNING) << "Generate Mask Failed" << pKF->mnId;
        return;
    }

    LOG(INFO) << "Generate Mask for Frame: " << pKF->mnId;

    auto start = std::chrono::steady_clock::now();

    cv::Mat mask = cv::Mat::zeros(pKF->mImLabel.size(), CV_8UC1);
    for (std::map<std::string, int>::iterator lit = mmDynamicObjects.begin(), lend = mmDynamicObjects.end(); lit != lend; lit++) {
        int label_id = lit->second;
        mask += (label_id == pKF->mImLabel);
    }

    // LOG(INFO) << "Size of mask: " << mask.size();
    // LOG(INFO) << "type of mask: " << mask.type();
    if (mask.empty()) {
        return;
    }
   
    mask.copyTo(pKF->mImMaskOld);

    // dilate the mask to filter out features on the edge of person and remove the noise of parts of body in PCD
    if (isDilate) {
        cv::dilate(mask, pKF->mImMask, mKernel);
        if (pKF->mImMask.empty()) {
            LOG(INFO) << "Dilate operation failed";
            return;
        }
    } else {
        pKF->mImMaskOld.copyTo(pKF->mImMask);
    }
    // pKF->InformSemanticReady(true);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    LOG(INFO) << "========Time to update moving probability:  " << std::setw(3) << diff.count() * 1000 << " ms";
    mvTimeMaskGeneration.push_back(diff.count());

    // save results
    // save original mask
    Config::GetInstance()->saveImage(pKF->mImMaskOld, "mask", "old_" + std::to_string(pKF->mnId) + ".png");
    // save dialated result
    if (isDilate) {
        Config::GetInstance()->saveImage(cv::Mat(pKF->mImMask).clone(), "mask", "dilate_" + std::to_string(pKF->mnId) + ".png");
    }
}

bool Semantic::IsInImage(const float& x, const float& y, const cv::Mat& img)
{
    return (x > 0 && x < img.cols && y > 0 && y < img.rows);
}

// 设置追踪线程句柄
void Semantic::SetTracker(Tracking* pTracker)
{
    mpTracker = pTracker;
    if (mpTracker) {
        LOG(INFO) << "========================Set tracker instance";
    }
}


Semantic::~Semantic()
{
    LOG(INFO) << "Deinit Semantic";
    if (mptSemanticSegmentation->joinable())
        mptSemanticSegmentation->join();

    if (mptSemanticTracking->joinable())
        mptSemanticTracking->join();
}

//得到二值Mask
void Semantic::getBinMask(const cv::Mat& comMask, cv::Mat& binMask)
{
    if (comMask.empty() || comMask.type() != CV_8UC1)
        //CV_Error(cv::Error::StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
    if (binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols)
        binMask.create(comMask.size(), CV_8UC1);
    binMask = comMask & 1;
}

Semantic* Semantic::mInstance = nullptr;  //???

}

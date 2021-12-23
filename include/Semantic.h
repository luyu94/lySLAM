#ifndef _SEMANTIC_H_
#define _SEMANTIC_H_

#include "Common.h"
// ORB SLAM
#include "KeyFrame.h"
#include "Tracking.h"


namespace ORB_SLAM2 {

class Semantic {
public:
    static Semantic* GetInstance();
    void IsEnableSemantic(bool in_isEnable);
    void SetTracker(Tracking* pTracker);

    void SetSemanticMethod(const std::string& cnn_method);

    // note: threads related
    void Run();
    void SemanticSegmentationThread();  //语义分割线程
    void SemanticTrackingThread();  //语义跟踪线程
    //void SemanticSegmentationThread();  //语义分割线程
    void SemanticBAThread();
    void RequestFinish();

    // keyframe queue management
    void InsertKeyFrame(KeyFrame* pKF);
    void InsertSemanticRequest(KeyFrame* pKF);
    size_t GetLatestSemanticKeyFrame();

    // Check the current map point whether is dynamic
    bool IsDynamicMapPoint(const MapPoint* pMP);
    static void getBinMask(const cv::Mat& comMask, cv::Mat& binMask);

    // Evaluation
    std::vector<float> mvTimeUpdateMovingProbability;
    std::vector<float> mvTimeMaskGeneration;
    std::vector<float> mvTimeSemanticOptimization;
    // To evaluate the time delay between sequential model and bi-direction model
    std::vector<size_t> mvSemanticDelay;

 

    void FinalStage();
    ~Semantic();


private:
    // latest keyframe that has semantic label
    size_t mnLatestSemanticKeyFrameID;
    // Total number of keyframes that has semantic label
    size_t mnTotalSemanticFrameNum;

    int mBatchSize;
    Tracking* mpTracker;
    Map* mpMap;

    std::string msCnnMethod;

    // Morphological filter
    int mDilation_size;
    cv::Mat mKernel;

    // disable or enable semantic moving probability
    bool mbIsUseSemantic;

    // segement the first few frames
    int mnSegmentFrameNum;

    // threshold of moving probability for map point
    float mthDynamicThreshold;

    // wait semantic result
    std::mutex mMutexResult;
    std::condition_variable mcvWaitResult;

    // priori dynamic objects
    std::map<std::string, int> mmDynamicObjects;

    // New semantic request
    bool CheckNewSemanticRequest();
    // inset keyframe
    bool CheckNewKeyFrames();
    bool CheckSemanticBARequest();

    bool CheckNewSemanticTrackRequest();
    void AddSemanticTrackRequest(KeyFrame* pKF);
    void AddSemanticBARequest(KeyFrame* pKF);
    void GenerateMask(KeyFrame* pKF, const bool isDilate = true);

    std::mutex mMutexNewSemanticRequest;
    std::list<KeyFrame*> mlNewSemanticRequest;

    KeyFrame* mpCurrentKeyFrame;
    std::list<KeyFrame*> mlNewKeyFrames;    //新的关键帧list

    std::mutex mMutexSemanticTrack;
    std::list<KeyFrame*> mlSemanticTrack;
    std::list<KeyFrame*> mlSemanticNew;
    std::list<KeyFrame*> mlSemanticBA;

    std::mutex mMutexNewKFs;
    std::mutex mMutexSemanticBA;

    bool IsInImage(const float& x, const float& y, const cv::Mat& img);

    // Finish Request
    bool CheckFinish();
    bool mbFinishRequested;     // 当前线程的主函数是否已经终止
    std::mutex mMutexFinish;    // 和"线程真正结束"有关的互斥锁

    // MaskRCNN
    cv::Mat maskRCNN;
    //maskrcnn_ros::MaskRCNN_Client_Batch::Ptr mMaskRCNN;
    //segnet_ros::SegNetClient::Ptr mSegNet;

    // semantic thread
    std::thread* mptSemanticSegmentation;
    std::thread* mptSemanticBA;
    // semantic tracking thread
    std::thread* mptSemanticTracking;

    static Semantic* mInstance;
    Semantic();
};
}
#endif

cmake_minimum_required(VERSION 2.8)
project(lySLAM)

IF(NOT CMAKE_BUILD_TYPE)
  #SET(CMAKE_BUILD_TYPE Release)
   SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -w -Wunused-parameter -O3 -march=native ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -w  -Wunused-parameter -O3 -march=native")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -W -Wunused-parameter -g -O0 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W -Wunused-parameter -g -O0 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHCi11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(OpenCV_DIR /usr/local/opencv2411/share/OpenCV)
message("OpenCV_DIR: " ${OpenCV_DIR})

#set(OpenCV_DIR /usr/local/opencv2411/share/OpenCV)
find_package(OpenCV 2.4.11 QUIET)
#if(NOT OpenCV_FOUND)
#   find_package(OpenCV 2.4.3 QUIET)
#   if(NOT OpenCV_FOUND)
#      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
#   endif()
#endif()

#find_package(Qt5Widgets REQUIRED)
#find_package(Qt5Concurrent REQUIRED)
#find_package(Qt5OpenGL REQUIRED)
#find_package(Qt5Test REQUIRED)

#set(PYTHON_LIBRARY "/root/anaconda3/envs/dyna/lib/libpython2.7.so")
set(PYTHON_LIBRARY "/usr/lib/x86_64-linux-gnu/libpython2.7.so")
find_package(PythonLibs REQUIRED)
if (NOT PythonLibs_FOUND)
    message(FATAL_ERROR "PYTHON LIBS not found.")
else()
    message("PYTHON LIBS were found!")
    message("PYTHON LIBS DIRECTORY: " ${PYTHON_LIBRARY})
endif()

find_package(Boost REQUIRED COMPONENTS thread)
if(Boost_FOUND)
    message("Boost was found!")
    message("Boost Headers DIRECTORY: " ${Boost_INCLUDE_DIRS})
    message("Boost LIBS DIRECTORY: " ${Boost_LIBRARY_DIRS})
    message("Found Libraries: " ${Boost_LIBRARIES})
endif()

set(EIGEN3_INCLUDE_DIR "/usr/local/include")
find_package(Eigen3 3.2.10 REQUIRED)
find_package(Pangolin REQUIRED)
#find_package(Glog REQUIRED)
#message("Glog_DIR: " ${Glog_DIR})
find_package(Boost REQUIRED COMPONENTS
  filesystem
)


#set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} /root/anaconda3/envs/dyna/lib/python2.7/site-packages/numpy/core/include/numpy)
set(PYTHON_INCLUDE_DIRS "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy")
message("PYTHON_INCLUDE_DIRS: " ${PYTHON_INCLUDE_DIRS})

###########
## Build ##
###########

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${GLOG_INCLUDE_DIRS}
/usr/include/python2.7/
/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/
#/root/anaconda3/envs/dyna/include/python2.7/
#/root/anaconda3/envs/dyna/lib/python2.7/site-packages/numpy/core/include/numpy/
${Boost_INCLUDE_DIRS}
)
message("PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/SlamConfig.cc   #
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
src/Conversion.cc
src/MaskNet.cc
src/Semantic.cc #
#src/Geometry.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
/usr/lib/x86_64-linux-gnu/libpython2.7.so
#/root/anaconda3/envs/dyna/lib/libpython2.7.so
${Boost_LIBRARIES}
-lboost_serialization
#-lcrypto
-lglog
)

# Build examples

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum_optical
Examples/RGB-D/rgbd_tum_optical.cc)
target_link_libraries(rgbd_tum_optical ${PROJECT_NAME})

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

#add_executable(stereo_kitti
#Examples/Stereo/stereo_kitti.cc)
#target_link_libraries(stereo_kitti ${PROJECT_NAME})

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

#add_executable(mono_tum
#Examples/Monocular/mono_tum.cc)
#target_link_libraries(mono_tum ${PROJECT_NAME})

#add_executable(mono_kitti
#Examples/Monocular/mono_kitti.cc)
#target_link_libraries(mono_kitti ${PROJECT_NAME})

#add_executable(mono_carla
#Examples/Monocular/mono_carla.cc)
#target_link_libraries(mono_carla ${PROJECT_NAME})


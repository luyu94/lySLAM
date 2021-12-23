echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

cd /mnt/DynaSLAM

echo "Configuring and building DynaSLAM ..."
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

#./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml /mnt/data/rgbd_dataset_freiburg3_walking_xyz /mnt/DynaSLAM/Examples/RGB-D/associations/fr3_walking_xyz.txt /mnt/masks /mnt/output
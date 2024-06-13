cmake_minimum_required(VERSION 3.8)
project(face-stereo)

set(CMAKE_CXX_STANDARD 17)

find_package (Eigen3 REQUIRED NO_MODULE)
find_package (OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/../calibr/include
	/usr/local/include/opencv4)

add_executable(face-stereo 
               ${PROJECT_SOURCE_DIR}/src/face_stereo.cpp
               ${PROJECT_SOURCE_DIR}/src/main_face_stereo_reconst.cpp)   

#add_library()

#add_library(lz4lib SHARED IMPORTED)
# set_property(TARGET lz4lib PROPERTY IMPORTED_LOCATION "/usr/lib64/liblz4.so")
	
target_link_libraries(face-stereo 
                    Eigen3::Eigen 
                    ${OpenCV_LIBS}
                    ${PROJECT_SOURCE_DIR}/../calibr/build/libcalibr.a)
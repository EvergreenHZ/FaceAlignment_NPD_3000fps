project(FaceAlignment)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp")
include(lib3000fps.cmake)

set(SRC src/main.cpp
        src/prepare.cpp
        src/test.cpp
        src/train.cpp
	src/npddetect.cpp
	src/npdmodel.cpp)

add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} lib3000fps ${OpenCV_LIBS})

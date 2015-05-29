cmake_minimum_required(VERSION 2.8.3)

project(drd_preprcessing)

find_package(OpenCV REQUIRED)

include_directories(. ${OpenCV_INCLUDE_DIRS})

add_library(iputils iputils/derivatives.cpp iputils/lcn.cpp iputils/trim.cpp)


add_executable(trimmer trimmer/trimmer.cpp)
target_link_libraries(trimmer iputils  ${OpenCV_LIBS})

add_executable(lcn lcn/lcn.cpp)
target_link_libraries(lcn iputils  ${OpenCV_LIBS})

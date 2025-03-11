cmake_minimum_required(VERSION 3.29)
project(lab1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "-O3 -march=native")

find_package(MPI REQUIRED)

set(SRC
        src/BinIO.cpp
        src/sleSolver.cpp
)

add_executable(${PROJECT_NAME} ${SRC} src/main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
        src/include
)
target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX)

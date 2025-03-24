# only for local testing
# see .sh scripts for running on cluster
cmake_minimum_required(VERSION 3.28)
project(lab1)

set(CMAKE_CXX_STANDARD 98)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -pedantic")
find_package(MPI REQUIRED)

set(SRC
        src/BinIO.cpp
        src/sleSolver.cpp
)

add_executable(${PROJECT_NAME} ${SRC} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX)

cmake_minimum_required(VERSION 3.29)
project(lab2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -march=native")

find_package(OpenMP REQUIRED)

set(SRC
        src/BinIO.cpp
        src/SLE_Solver.cpp
        src/solvers/NonParallel.cpp
        src/solvers/MultipleSectionsOMP.cpp
        src/solvers/OneSectionOMP.cpp
)

add_executable(${PROJECT_NAME} ${SRC} src/main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
        src/include
        src/include/solvers
)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

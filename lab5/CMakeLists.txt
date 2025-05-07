cmake_minimum_required(VERSION 3.28)
project(lab5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -pedantic")
find_package(MPI REQUIRED)

set(SRC
        src/TaskManager.cpp
)

add_executable(${PROJECT_NAME} ${SRC} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX)

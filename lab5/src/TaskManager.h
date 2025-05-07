#pragma once

#include <mpi.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

class TaskManager {
    enum QueueFill { BALANCED, MODERATE, UNBALANCED };

    static constexpr QueueFill QUEUE_FILL = MODERATE;
    static constexpr bool BALANCING_ENABLED = true;
    static constexpr int TOTAL_TASK_COUNT = 100;
    static constexpr int BASE_WEIGHT = 200;  // ms
    static constexpr int BALANCED_FACTOR = 4;
    static constexpr int UNBALANCED_FACTOR = 4;

    static constexpr int REQUEST_TAG = 0;
    static constexpr int RESPONSE_TAG = 1;
    static constexpr int EMPTY_QUEUE_RESPONSE = -1;
    static constexpr int FINISH_TAG = -1;
    MPI_Comm COMM;

    class ThreadSafeQueue {
     public:
        void push(int value);
        int pop();
        bool empty();
        size_t getSize();
        void setFinish();

     private:
        std::queue<int> data;
        std::mutex qMutex;
        std::condition_variable qCondVar;
        bool qFinish = false;
    };

    int procID, commSize;
    std::mutex rMutex;
    std::condition_variable rCondVar;
    bool rFinish = false;

    ThreadSafeQueue taskQueue;
    std::thread workerThread;
    std::thread receiverThread;
    std::thread senderThread;

    void workerRun();
    void receiverRun();
    void senderRun();

    void fillBalanced();
    void fillModerate();
    void fillUnbalanced();

 public:
    explicit TaskManager(MPI_Comm comm);
    void run();
};

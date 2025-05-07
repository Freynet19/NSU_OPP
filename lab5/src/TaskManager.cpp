#include "TaskManager.h"
#include <iostream>
#include <chrono>

void TaskManager::ThreadSafeQueue::push(int value) {
    std::lock_guard lock(qMutex);
    data.push(value);
    qCondVar.notify_one();
}

int TaskManager::ThreadSafeQueue::pop() {
    if constexpr (BALANCING_ENABLED) {
        std::unique_lock lock(qMutex);
        qCondVar.wait(lock, [this]{ return !data.empty() || qFinish; });

        if (qFinish) return FINISH_TAG;

        const int value = data.front();
        data.pop();
        return value;
    } else {
        const int value = data.front();
        data.pop();
        return value;
    }
}

bool TaskManager::ThreadSafeQueue::empty() {
    std::lock_guard lock(qMutex);
    return data.empty();
}

size_t TaskManager::ThreadSafeQueue::getSize() {
    std::lock_guard lock(qMutex);
    return data.size();
}

void TaskManager::ThreadSafeQueue::setFinish() {
    std::lock_guard lock(qMutex);
    qFinish = true;
    qCondVar.notify_all();
}

void TaskManager::workerRun() {
    while (true) {
        if (taskQueue.empty()) {
            if constexpr (BALANCING_ENABLED) {
                std::lock_guard lock(rMutex);
                rCondVar.notify_one();
            } else {
                break;
            }
        }

        int weight = taskQueue.pop();
        if (weight == FINISH_TAG) break;

        if (taskQueue.empty()) {
            if constexpr (BALANCING_ENABLED) {
                std::lock_guard lock(rMutex);
                rCondVar.notify_one();
            } else {
                break;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(weight));
        std::cout << "Worker " << procID << " slept for " << weight <<
            "ms, tasks left: " << taskQueue.getSize() << std::endl;
    }
    std::cout << "Worker " << procID << " finished" << std::endl;
}

void TaskManager::receiverRun() {
    while (true) {
        if (!taskQueue.empty()) {
            std::unique_lock lock(rMutex);
            rCondVar.wait(lock, [this]
                { return taskQueue.empty() || rFinish; });
        }
        if (rFinish) break;

        int recvCount = 0;
        for (int sendrecvID = 0; sendrecvID < commSize; ++sendrecvID) {
            if (sendrecvID == procID) continue;
            MPI_Send(&procID, 1, MPI_INT, sendrecvID, REQUEST_TAG, COMM);
            // std::cout << "Receiver " << procID << " sent request " <<
            //         "to sender " << sendrecvID << std::endl;

            MPI_Request req;
            int weight, isReceived;
            MPI_Irecv(&weight, 1, MPI_INT, sendrecvID, RESPONSE_TAG, COMM,
                &req);

            do {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                MPI_Test(&req, &isReceived, MPI_STATUS_IGNORE);
            } while (!isReceived && !rFinish);

            if (rFinish) break;

            if (weight != EMPTY_QUEUE_RESPONSE) {
                recvCount++;
                taskQueue.push(weight);
                std::cout << "Receiver " << procID << " received " << weight <<
                    "ms task from sender " << sendrecvID << std::endl;
            } else {
                // std::cout << "Receiver " << procID << " received empty queue "
                //     "response from sender " << sendrecvID << std::endl;
            }
        }
        if (recvCount == 0) break;
    }
    taskQueue.setFinish();  // makes own worker finish
    for (int finID = 0; finID < commSize; ++finID) {
        MPI_Send(&FINISH_TAG, 1, MPI_INT, finID, REQUEST_TAG, COMM);
        // makes all senders finish
    }
    std::cout << "Receiver " << procID << " finished" << std::endl;
}

void TaskManager::senderRun() {
    int sendrecvID;
    while (true) {
        MPI_Recv(&sendrecvID, 1, MPI_INT, MPI_ANY_SOURCE,
            REQUEST_TAG, COMM, MPI_STATUS_IGNORE);
        if (sendrecvID == FINISH_TAG) break;
        // std::cout << "Sender " << procID << " received "
        //     "request from receiver " << sendrecvID << std::endl;

        if (!taskQueue.empty()) {
            int weight = taskQueue.pop();
            MPI_Send(&weight, 1, MPI_INT, sendrecvID,
                RESPONSE_TAG, COMM);
            // std::cout << "Sender " << procID << " sent " << weight <<
            //     "ms task to receiver " << sendrecvID << std::endl;
        } else {
            MPI_Send(&EMPTY_QUEUE_RESPONSE, 1, MPI_INT, sendrecvID,
                RESPONSE_TAG, COMM);
            // std::cout << "Sender " << procID << " sent empty queue "
            //     "response to receiver " << sendrecvID << std::endl;
        }
    }
    {
        std::unique_lock lock(rMutex);
        rFinish = true;  // makes own receiver finish
        rCondVar.notify_all();
    }
    std::cout << "Sender " << procID << " finished" << std::endl;
}

void TaskManager::fillBalanced() {
    const int taskCount = TOTAL_TASK_COUNT / commSize +
        (procID < TOTAL_TASK_COUNT % commSize);
    for (int i = 0; i < taskCount; ++i) {
        taskQueue.push(BASE_WEIGHT * BALANCED_FACTOR);
    }
    if (procID == 0) std::cout << "Generated balanced tasks" << std::endl;
}

void TaskManager::fillModerate() {
    const int taskCount = TOTAL_TASK_COUNT / commSize +
        (procID < TOTAL_TASK_COUNT % commSize);
    for (int i = 0; i < taskCount; ++i) {
        const int factor = std::abs(i % commSize - procID) + 1;
        taskQueue.push(BASE_WEIGHT * factor * factor);
    }
    if (procID == 0) std::cout << "Generated moderate tasks" << std::endl;
}

void TaskManager::fillUnbalanced() {
    if (procID == 0) {
        for (int i = 0; i < TOTAL_TASK_COUNT; ++i) {
            taskQueue.push(BASE_WEIGHT * UNBALANCED_FACTOR);
        }
        std::cout << "Generated unbalanced tasks" << std::endl;
    }
}

TaskManager::TaskManager(MPI_Comm comm): COMM(comm) {
    MPI_Comm_rank(COMM, &procID);
    MPI_Comm_size(COMM, &commSize);

    switch (QUEUE_FILL) {
        case BALANCED: fillBalanced(); break;
        case MODERATE: fillModerate(); break;
        case UNBALANCED: fillUnbalanced(); break;
    }

    if (procID == 0) {
        std::cout << "Total tasks: " << TOTAL_TASK_COUNT << std::endl;
        std::cout << "Base task weight: " << BASE_WEIGHT << std::endl;
    }
}

void TaskManager::run() {
    MPI_Barrier(COMM);
    const double start = MPI_Wtime();

    workerThread = std::thread(&TaskManager::workerRun, this);
    if constexpr (BALANCING_ENABLED) {
        receiverThread = std::thread(&TaskManager::receiverRun, this);
        senderThread = std::thread(&TaskManager::senderRun, this);
        if (procID == 0) std::cout << "Running w/ balancing..." << std::endl;
    } else {
        if (procID == 0) std::cout << "Running w/o balancing..." << std::endl;
    }

    workerThread.join();
    if constexpr (BALANCING_ENABLED) {
        receiverThread.join();
        senderThread.join();
    }

    std::cout << "Process " << procID << " finished" << std::endl;

    MPI_Barrier(COMM);
    const double end = MPI_Wtime();
    if (procID == 0) std::cout << "Time taken: " << end - start << std::endl;
}

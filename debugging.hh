#ifndef DEBUGGING_HH_INCLUDED
#define DEBUGGING_HH_INCLUDED

#ifndef DEBUGGING
#define DEBUGGING 1
#endif

#if DEBUGGING


#endif

#if DEBUGGING

#include <iostream>
#include <mutex>

#include "filter.hh"

static std::mutex debugging_mtx;

#define DEBUG_RAW_CODE(X) X
#define DEBUG_CODE(X) do { std::unique_lock<std::mutex> lock(debugging_mtx); do { X } while (0); } while (0)

#define DEBUG_PRINT(X) do { std::unique_lock<std::mutex> lock(debugging_mtx); do { std::cout << X ; } while (0); } while (0)
#define DEBUG_PRINTLN(X) do { std::unique_lock<std::mutex> lock(debugging_mtx); do { std::cout << X << std::endl; } while (0); } while (0)

#else
#define DEBUG_RAW_CODE(X)
#define DEBUG_PRINT(X)
#define DEBUG_PRINTLN(X)
#define DEBUG_CODE(X)
#endif

#endif

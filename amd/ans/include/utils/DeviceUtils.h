/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <iostream>
#include <hip/hip_runtime.h>
#include <string>
#include <stdexcept>
#include <vector>

// #include "DeviceUtils0.h"
#include <hip/hip_runtime_api.h>
#include <mutex>
#include <unordered_map>

/// Wrapper to test return status of CUDA functions
#define CUDA_VERIFY(X)                                                   \
  do {                                                                   \
    auto err__ = (X);                                                    \
    if (err__ != hipSuccess) {                                          \
      throw std::runtime_error(                                          \
          std::string(#X) + ": " + pans_hip::errorToName(err__) + " " + \
          pans_hip::errorToString(err__));                               \
    }                                                                    \
  } while (0)

/// Wrapper to synchronously probe for CUDA errors
// #define GPU_SYNC_ERROR 1

#ifdef GPU_SYNC_ERROR
#define CUDA_TEST_ERROR()                 \
  do {                                    \
    CUDA_VERIFY(hipDeviceSynchronize()); \
  } while (0)
#else
#define CUDA_TEST_ERROR()            \
  do {                               \
    CUDA_VERIFY(hipGetLastError()); \
  } while (0)
#endif

namespace pans_hip{
  
constexpr int kWarpSize = 64;

/// std::string wrapper around hipGetErrorString
std::string errorToString(hipError_t err);

/// std::string wrapper around hipGetErrorName
std::string errorToName(hipError_t err);

/// Returns the current thread-local GPU device
int getCurrentDevice();

/// Sets the current thread-local GPU device
void setCurrentDevice(int device);

/// Returns the number of available GPU devices
int getNumDevices();

/// Starts the CUDA profiler (exposed via SWIG)
void profilerStart();

/// Stops the CUDA profiler (exposed via SWIG)
void profilerStop();

/// Synchronizes the CPU against all devices (equivalent to
/// hipDeviceSynchronize for each device)
void synchronizeAllDevices();

/// Returns a cached hipDeviceProp_t for the given device
const hipDeviceProp_t& getDeviceProperties(int device);

/// Returns the cached hipDeviceProp_t for the current device
const hipDeviceProp_t& getCurrentDeviceProperties();

/// Returns the maximum number of threads available for the given GPU
/// device
int getMaxThreads(int device);

/// Equivalent to getMaxThreads(getCurrentDevice())
int getMaxThreadsCurrentDevice();

/// Returns the maximum smem available for the given GPU device
size_t getMaxSharedMemPerBlock(int device);

/// Equivalent to getMaxSharedMemPerBlock(getCurrentDevice())
size_t getMaxSharedMemPerBlockCurrentDevice();

/// For a given pointer, returns whether or not it is located on
/// a device (deviceId >= 0) or the host (-1).
int getDeviceForAddress(const void* p);

/// Does the given device support full unified memory sharing host
/// memory?
bool getFullUnifiedMemSupport(int device);

/// Equivalent to getFullUnifiedMemSupport(getCurrentDevice())
bool getFullUnifiedMemSupportCurrentDevice();

/// RAII object to set the current device, and restore the previous
/// device upon destruction
class DeviceScope {
 public:
  explicit DeviceScope(int device);
  ~DeviceScope();

 private:
  int prevDevice_;
};

// RAII object to manage a hipEvent_t
class CudaEvent {
 public:
  /// Creates an event and records it in this stream
  explicit CudaEvent(hipStream_t stream, bool timer = false);
  CudaEvent(const CudaEvent& event) = delete;
  CudaEvent(CudaEvent&& event) noexcept;
  ~CudaEvent();

  CudaEvent& operator=(CudaEvent&& event) noexcept;
  CudaEvent& operator=(CudaEvent& event) = delete;

  inline hipEvent_t get() {
    return event_;
  }

  /// Wait on this event in this stream
  void streamWaitOnEvent(hipStream_t stream);

  /// Have the CPU wait for the completion of this event
  void cpuWaitOnEvent();

  /// Returns the elapsed time from the other event
  float timeFrom(CudaEvent& from);

 private:
  hipEvent_t event_;
};

// RAII object to manage a hipStream_t
class CudaStream {
 public:
  /// Creates a stream on the current device
  CudaStream(int flags = hipStreamDefault);
  CudaStream(const CudaStream& stream) = delete;
  CudaStream(CudaStream&& stream) noexcept;
  ~CudaStream();

  CudaStream& operator=(CudaStream&& stream) noexcept;
  CudaStream& operator=(CudaStream& stream) = delete;

  inline hipStream_t get() {
    return stream_;
  }

  operator hipStream_t() {
    return stream_;
  }

  static CudaStream make();
  static CudaStream makeNonBlocking();

 private:
  hipStream_t stream_;
};

/// Call for a collection of streams to wait on
template <typename L1, typename L2>
void streamWaitBase(const L1& listWaiting, const L2& listWaitOn) {
  // For all the streams we are waiting on, create an event
  std::vector<hipEvent_t> events;
  for (auto& stream : listWaitOn) {
    hipEvent_t event;
    CUDA_VERIFY(hipEventCreateWithFlags(&event, hipEventDisableTiming));
    CUDA_VERIFY(hipEventRecord(event, stream));
    events.push_back(event);
  }

  // For all the streams that are waiting, issue a wait
  for (auto& stream : listWaiting) {
    for (auto& event : events) {
      CUDA_VERIFY(hipStreamWaitEvent(stream, event, 0));
    }
  }

  for (auto& event : events) {
    CUDA_VERIFY(hipEventDestroy(event));
  }
  return;
}

/// These versions allow usage of initializer_list as arguments, since
/// otherwise {...} doesn't have a type
template <typename L1>
void streamWait(const L1& a, const std::initializer_list<hipStream_t>& b) {
  streamWaitBase(a, b);
  return;
}

template <typename L2>
void streamWait(const std::initializer_list<hipStream_t>& a, const L2& b) {
  streamWaitBase(a, b);
  return;
}

inline void streamWait(
    const std::initializer_list<hipStream_t>& a,
    const std::initializer_list<hipStream_t>& b) {
  streamWaitBase(a, b);
  return;
}

std::string errorToString(hipError_t err) {
  return std::string(hipGetErrorString(err));
}

std::string errorToName(hipError_t err) {
  return std::string(hipGetErrorName(err));
}

int getCurrentDevice() {
  int dev = -1;
  CUDA_VERIFY(hipGetDevice(&dev));

  return dev;
}

void setCurrentDevice(int device) {
  CUDA_VERIFY(hipSetDevice(device));
  return;
}

int getNumDevices() {
  int numDev = -1;
  hipError_t err = hipGetDeviceCount(&numDev);
  if (hipErrorNoDevice == err) {
    numDev = 0;
  } else {
    CUDA_VERIFY(err);
  }

  return numDev;
}

void profilerStart() {
  // CUDA_VERIFY(hipProfilerStart());
  return;
}

void profilerStop() {
  // CUDA_VERIFY(hipProfilerStop());
  return;
}

void synchronizeAllDevices() {
  for (int i = 0; i < getNumDevices(); ++i) {
    DeviceScope scope(i);

    CUDA_VERIFY(hipDeviceSynchronize());
  }
  return;
}

const hipDeviceProp_t& getDeviceProperties(int device) {
  static std::mutex mutex;
  static std::unordered_map<int, hipDeviceProp_t> properties;

  std::lock_guard<std::mutex> guard(mutex);

  auto it = properties.find(device);
  if (it == properties.end()) {
    hipDeviceProp_t prop;
    CUDA_VERIFY(hipGetDeviceProperties(&prop, device));

    properties[device] = prop;
    it = properties.find(device);
  }

  return it->second;
}

const hipDeviceProp_t& getCurrentDeviceProperties() {
  return getDeviceProperties(getCurrentDevice());
}

int getMaxThreads(int device) {
  return getDeviceProperties(device).maxThreadsPerBlock;
}

int getMaxThreadsCurrentDevice() {
  return getMaxThreads(getCurrentDevice());
}

size_t getMaxSharedMemPerBlock(int device) {
  return getDeviceProperties(device).sharedMemPerBlock;
}

size_t getMaxSharedMemPerBlockCurrentDevice() {
  return getMaxSharedMemPerBlock(getCurrentDevice());
}

int getDeviceForAddress(const void* p) {
  if (!p) {
    return -1;
  }

  hipPointerAttribute_t att;
  hipError_t err = hipPointerGetAttributes(&att, p);

  if (err == hipErrorInvalidValue) {
    // Make sure the current thread error status has been reset
    err = hipGetLastError();

    return -1;
  }

  // memoryType is deprecated for CUDA 10.0+
#if defined(CUDA_VERSION) && CUDA_VERSION < 10000
  if (att.memoryType == hipMemoryTypeHost) {
    return -1;
  } else {
    return att.device;
  }
#else
  // FIXME: what to use for managed memory?
  if (att.type == hipMemoryTypeDevice) {
    return att.device;
  } else {
    return -1;
  }
#endif
}

bool getFullUnifiedMemSupport(int device) {
  const auto& prop = getDeviceProperties(device);
  return (prop.major >= 6);
}

bool getFullUnifiedMemSupportCurrentDevice() {
  return getFullUnifiedMemSupport(getCurrentDevice());
}

DeviceScope::DeviceScope(int device) {
  if (device >= 0) {
    int curDevice = getCurrentDevice();

    if (curDevice != device) {
      prevDevice_ = curDevice;
      setCurrentDevice(device);
      return;
    }
  }

  // Otherwise, we keep the current device
  prevDevice_ = -1;
}

DeviceScope::~DeviceScope() {
  if (prevDevice_ != -1) {
    setCurrentDevice(prevDevice_);
  }
}

// CudaEvent::CudaEvent(hipStream_t stream, bool timer) : event_(nullptr) {
//   CUDA_VERIFY(hipEventCreateWithFlags(
//       &event_, timer ? hipEventDefault : hipEventDisableTiming));
//   CUDA_VERIFY(hipEventRecord(event_, stream));
// }

// CudaEvent::CudaEvent(CudaEvent&& event) noexcept
//     : event_(std::move(event.event_)) {
//   event.event_ = nullptr;
// }

// CudaEvent::~CudaEvent() {
//   if (event_) {
//     CUDA_VERIFY(hipEventDestroy(event_));
//   }
// }

// CudaEvent& CudaEvent::operator=(CudaEvent&& event) noexcept {
//   event_ = std::move(event.event_);
//   event.event_ = nullptr;

//   return *this;
// }

// void CudaEvent::streamWaitOnEvent(hipStream_t stream) {
//   CUDA_VERIFY(hipStreamWaitEvent(stream, event_, 0));
// }

// void CudaEvent::cpuWaitOnEvent() {
//   CUDA_VERIFY(hipEventSynchronize(event_));
// }

// float CudaEvent::timeFrom(CudaEvent& from) {
//   cpuWaitOnEvent();
//   float ms = 0;
//   CUDA_VERIFY(hipEventElapsedTime(&ms, from.event_, event_));

//   return ms;
// }

// CudaStream::CudaStream(int flags) : stream_(nullptr) {
//   CUDA_VERIFY(hipStreamCreateWithFlags(&stream_, flags));
// }

// CudaStream::CudaStream(CudaStream&& stream) noexcept
//     : stream_(std::move(stream.stream_)) {
//   stream.stream_ = nullptr;
// }

// CudaStream::~CudaStream() {
//   if (stream_) {
//     CUDA_VERIFY(hipStreamDestroy(stream_));
//   }
// }

// CudaStream& CudaStream::operator=(CudaStream&& stream) noexcept {
//   stream_ = std::move(stream.stream_);
//   stream.stream_ = nullptr;

//   return *this;
// }

// CudaStream CudaStream::make() {
//   return CudaStream();
// }

// CudaStream CudaStream::makeNonBlocking() {
//   return CudaStream(hipStreamNonBlocking);
// }

}

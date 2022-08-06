#pragma once

#include <cinttypes>
#include <mutex>
#include <atomic>

namespace matcha::engine {

class RefReqCounted {
public:
  explicit RefReqCounted();
  virtual ~RefReqCounted();

//  void ref();
//  void unref();
  void req();
  void unreq();

//  unsigned refs() const;
  unsigned reqs() const;

private:
//  std::atomic_uint refs_;
//  std::atomic_uint reqs_;
//  std::mutex mtx_;

  unsigned refs_;
  unsigned reqs_;
};

}
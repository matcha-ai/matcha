#pragma once

#include "bits_of_matcha/engine/op/Op.h"

#include <iostream>


namespace matcha::engine::ops {

struct LoadImage : Op {
  LoadImage(const std::string& file);
  LoadImage(const std::string& file, const Frame& frame);

  void run() override;

private:
  std::string file_;

  Frame getFramePng(FILE* fp);
  Frame getFrameJpeg(FILE* fp);
  void dumpDataPng(FILE* fp);
  void dumpDataJpeg(FILE* fp);

  enum {
    Png, Jpeg
  } type_;

};

}
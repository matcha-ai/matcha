#include "bits_of_matcha/engine/ops/LoadImage.h"
#include "bits_of_matcha/engine/tensor/iterations.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <execution>
#include <png.h>

namespace matcha::engine::ops {

LoadImage::LoadImage(const std::string& file)
  : Op{}
  , file_(file)
{
  FILE* fp = fopen(file.c_str(), "r");
  if (!fp) throw std::runtime_error("can't open given .png file");
  outputs.add(this, getFrame(fp));
  fclose(fp);
}

LoadImage::LoadImage(const std::string& file, const Frame& frame)
  : Op{}
  , file_(file)
{
  outputs.add(this, frame);
}

void LoadImage::run() {
  FILE* fp = fopen(file_.c_str(), "r");
  dumpData(fp);
  fclose(fp);
}

Frame LoadImage::getFrame(FILE* fp) {
  png_structp pngPtr = png_create_read_struct(
    PNG_LIBPNG_VER_STRING,
    nullptr,
    nullptr,
    nullptr
  );
  png_infop infoPtr = png_create_info_struct(pngPtr);
  png_init_io(pngPtr, fp);
  uint32_t width, height;
  unsigned channels = 3;
  int bitDepth, colorType;
  int interface, compression, filter;
  png_read_info(pngPtr, infoPtr);
  png_get_IHDR(
    pngPtr, infoPtr,
    &width, &height,
    &bitDepth, &colorType,
    &interface, &compression, &filter
  );
  png_destroy_read_struct(&pngPtr, &infoPtr, nullptr);
  return {Float, {channels, (unsigned) height, (unsigned) width}};
}

void LoadImage::dumpData(FILE* fp) {
  png_structp pngPtr = png_create_read_struct(
    PNG_LIBPNG_VER_STRING,
    nullptr,
    nullptr,
    nullptr
  );
  png_infop infoPtr = png_create_info_struct(pngPtr);
  png_init_io(pngPtr, fp);
  uint32_t width, height;
  unsigned channels = 3;
  int bitDepth, colorType;
  int interface, compression, filter;
  png_read_info(pngPtr, infoPtr);
  png_get_IHDR(
    pngPtr, infoPtr,
    &width, &height,
    &bitDepth, &colorType,
    &interface, &compression, &filter
  );

  Shape shape = {channels, (unsigned) height, (unsigned) width};

  auto t = outputs[0];
  auto b = t->malloc().as<float*>();

  if (t->shape() != shape)
    throw std::runtime_error("image dimensions don't match the expected tensor shape");

  MatrixStackIteration iter(t->shape());

  png_bytep rowPtrs[height];
  size_t bytesPerRow = png_get_rowbytes(pngPtr, infoPtr);
  for (auto& rp: rowPtrs)
    rp = new png_byte[bytesPerRow];

  png_read_image(pngPtr, rowPtrs);

  float* tensorEnd = b + t->size();
  size_t c = 0;
  for (float* channel = b; channel != tensorEnd; channel += iter.size) {
    float* channelEnd = channel + iter.size;
    *channel = 3;
    size_t y = 0;
    for (float* row = channel; row != channelEnd; row += iter.cols) {
      png_bytep rowPtr = rowPtrs[y] + c;
      float* rowEnd = row + iter.cols;
      for (float* col = row; col != rowEnd; col++) {
        *col = *rowPtr;
        rowPtr += 3;
      }
      y++;
    }
    c++;
  }

  for (auto& rp: rowPtrs)
    delete[] rp;

  png_read_end(pngPtr, nullptr);
  png_destroy_read_struct(&pngPtr, &infoPtr, nullptr);
}

}
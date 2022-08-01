#include "bits_of_matcha/engine/ops/SaveImage.h"
#include "bits_of_matcha/engine/tensor/iterations.h"

#include <png.h>
#include <algorithm>
#include <numeric>
#include <execution>


namespace matcha::engine::ops {

SaveImage::SaveImage(Tensor* a, const std::string& file)
  : Op{a}
  , file_(file)
{}

Reflection<SaveImage> SaveImage::reflection {
  .name = "SaveImage",
  .side_effect = true,
};

void SaveImage::run() {
  dumpPng();
}

void SaveImage::dumpPng() {
  FILE *fp = fopen(file_.c_str(), "wb");
  if (!fp) throw std::runtime_error("couldn't open file for writing image");

  png_structp pngPtr = png_create_write_struct(
    PNG_LIBPNG_VER_STRING,
    (png_voidp) nullptr,
    nullptr,
    nullptr
  );
  if (!pngPtr) throw std::runtime_error("png ptr is null");

  png_infop infoPtr = png_create_info_struct(pngPtr);
  if (!infoPtr) throw std::runtime_error("info ptr is null");

  if (setjmp(png_jmpbuf(pngPtr))) {
    throw std::runtime_error("setjmp failed");
  }

  Tensor* a = inputs[0];
  auto buffer = a->buffer().as<float*>();

  auto [minEl, maxEl] = std::minmax_element(buffer, buffer + a->size());
  float min = *minEl, max = *maxEl;
  float range = max - min;

  auto normalizeValue = [=] (float value) {
    return (value - min) / range;
  };

  MatrixStackIteration iter(a->shape());
  std::string title = a->frame().string();

  png_init_io(pngPtr, fp);
  png_set_IHDR(
    pngPtr,
    infoPtr,
    iter.cols,
    iter.rows * iter.amount,
    8,
    PNG_COLOR_TYPE_RGB,
    PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_BASE,
    PNG_FILTER_TYPE_BASE
  );

  png_write_info(pngPtr, infoPtr);

  auto row_buffer = new png_byte[3 * iter.cols];

  for (int matrix = 0; matrix < iter.amount; matrix++) {
    for (int row = 0; row < iter.rows; row++) {
      for (int col = 0; col < iter.cols; col++) {
        float value = buffer[matrix * iter.size + row * iter.cols + col];
        uint8_t& r = row_buffer[3 * col];
        uint8_t& g = row_buffer[3 * col + 1];
        uint8_t& b = row_buffer[3 * col + 2];

        auto color = (uint8_t) (256 * normalizeValue(value));
        r = g = b = color;
      }
      png_write_row(pngPtr, row_buffer);
    }
  }

  png_write_end(pngPtr, nullptr);
  if (fp) fclose(fp);
  if (infoPtr) png_free_data(pngPtr, infoPtr, PNG_FREE_ALL, -1);
  delete[] row_buffer;
}

}
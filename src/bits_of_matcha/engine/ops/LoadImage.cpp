#include "bits_of_matcha/engine/ops/LoadImage.h"
#include "bits_of_matcha/engine/tensor/iterations.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <execution>
#include <png.h>
#include <jpeglib.h>

namespace matcha::engine::ops {

LoadImage::LoadImage(const std::string& file)
  : Op{}
  , file_(file)
{
  FILE* fp = fopen(file.c_str(), "r");
  if (!fp) throw std::runtime_error("can't open given image file");

  if (file_.ends_with(".png") || file_.ends_with(".PNG")) {
    type_ = Png;
  } else if (file_.ends_with(".jpg") || file_.ends_with(".JPG") ||
      file_.ends_with(".jpeg") || file_.ends_with(".JPEG")) {
    type_ = Jpeg;
  } else {
    throw std::runtime_error("unsupported image format");
  }

  addOutput(type_ == Png ? getFramePng(fp) : getFrameJpeg(fp));
  fclose(fp);
}

LoadImage::LoadImage(const std::string& file, const Frame& frame)
  : Op{}
  , file_(file)
{
  addOutput(frame);
}

void LoadImage::run() {
  FILE* fp = fopen(file_.c_str(), "r");
  switch (type_) {
  case Png: dumpDataPng(fp); break;
  case Jpeg: dumpDataJpeg(fp); break;
  }
  fclose(fp);
}

Frame LoadImage::getFramePng(FILE* fp) {
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
  return {Int, {channels, (unsigned) height, (unsigned) width}};
}

void LoadImage::dumpDataPng(FILE* fp) {
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
  auto b = t->malloc().as<int32_t*>();

  if (t->shape() != shape)
    throw std::runtime_error("image dimensions don't match the expected tensor shape");

  MatrixStackIteration iter(t->shape());

  png_bytep rowPtrs[height];
  size_t bytesPerRow = png_get_rowbytes(pngPtr, infoPtr);
  for (auto& rp: rowPtrs)
    rp = new png_byte[bytesPerRow];

  png_read_image(pngPtr, rowPtrs);

  int32_t* tensor_end = b + t->size();
  size_t c = 0;
  for (int32_t* channel = b; channel != tensor_end; channel += iter.size) {
    int32_t* channelEnd = channel + iter.size;
    *channel = 3;
    size_t y = 0;
    for (int32_t* row = channel; row != channelEnd; row += iter.cols) {
      png_bytep rowPtr = rowPtrs[y] + c;
      int32_t* row_end = row + iter.cols;
      for (int32_t* col = row; col != row_end; col++) {
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

Frame LoadImage::getFrameJpeg(FILE* fp) {
  jpeg_decompress_struct cinfo;
  jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, fp);
  jpeg_read_header(&cinfo, true);
  Frame frame(Int, {
    (unsigned) cinfo.num_components,
    cinfo.image_height,
    cinfo.image_width}
    );
  jpeg_destroy_decompress(&cinfo);

  return frame;
}

void LoadImage::dumpDataJpeg(FILE* fp) {
  auto t = outputs[0];
  auto b = t->malloc().as<int32_t*>();

  jpeg_decompress_struct cinfo;
  jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, fp);
  jpeg_read_header(&cinfo, true);
  if (cinfo.num_components != t->shape()[0] ||
      cinfo.image_height != t->shape()[-2] ||
      cinfo.image_width != t->shape()[-1])
    throw std::runtime_error("image dimensions don't match tensor shape");

  jpeg_start_decompress(&cinfo);
  JSAMPARRAY buffer;
  int row_stride = cinfo.image_width * cinfo.output_components;

  buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

  auto iter = MatrixStackIteration(t->shape());
  std::vector<int32_t*> channels(iter.amount);
  channels[0] = b;
  for (int i = 1; i < channels.size(); i++)
    channels[i] = channels[i - 1] + iter.size;

  if (t->shape()[0] != 3)
    throw std::runtime_error("currently only RGB format is supported");

  while (cinfo.output_scanline < cinfo.output_height) {
    jpeg_read_scanlines(&cinfo, buffer, 1);

    uint8_t* byte = (uint8_t*) buffer[0];
    for (size_t x = 0; x < iter.cols; x++) {
      for (auto& c: channels) {
        *c++ = *byte++;
      }
    }
  }
  jpeg_finish_decompress(&cinfo);

  jpeg_destroy_decompress(&cinfo);
}

}
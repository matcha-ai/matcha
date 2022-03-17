#pragma once

#include "bits_of_matcha/data/Dataset.h"

#include <string>
#include <vector>
#include <fstream>


namespace matcha::dataset {

class Csv {
public:
  const std::string file;
  const char delimiter = ',';
  const std::vector<std::string> y_cols = {"y", "label", "target", "class", "category", "result"};

private:
  class Internal : public Dataset::Internal {
  public:
    explicit Internal(Csv* csv);
    void seek(size_t pos) override;
    size_t tell() const override;
    size_t size() const override;
    Instance get() override;

  private:
    std::ifstream is_;
    size_t pos_;
    size_t size_;
    std::vector<std::string> rowBuffer_;
    std::vector<std::string> head_;
    std::vector<unsigned> colsX_;
    std::vector<unsigned> colsY_;
    void readRow(std::vector<std::string>& buffer);
    void findColsXY(std::vector<std::string> y);
    engine::Tensor* getX();
    engine::Tensor* getY();
    size_t skipRows(size_t amount);
  };

private:
  Internal* init();
  MA_DATASET_TAIL();
};

}
#include "bits_of_matcha/dataset/loaders/Csv.h"
#include "bits_of_matcha/engine/dataset/Dataset.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/print.h"

#include <fstream>
#include <sstream>
#include <utility>


namespace matcha::dataset {

Csv::operator Dataset() {
  struct Internal : engine::Dataset {
    std::string file_;
    size_t size_;
    size_t pos_;
    std::ifstream is_;
    std::vector<std::string> cache_;
    std::vector<int> indicesX_, indicesY_;
    std::set<std::string> classification_tags_;
    std::set<std::string> regression_tags_;
    bool regression_;

    Internal(const std::string& file, std::set<std::string>  classification_tags, std::set<std::string>  regression_tags)
      : file_(file)
      , classification_tags_(std::move(classification_tags))
      , regression_tags_(std::move(regression_tags))
      , size_(-1)
      , pos_(0)
    {
      is_.open(file);
      if(!is_.is_open()) {
        throw std::runtime_error("couldn't open file");
      }
      cacheLine();
      findIndices();
      findSize();
      pos_ = -1;
      seekInternal(0);
    }

    void findIndices() {
      for (int i = 0; i < cache_.size(); i++) {
        auto& head = cache_[i];
        if (classification_tags_.contains(head)) {
          if (regression_ && !indicesY_.empty())
            throw std::runtime_error("currently Csv supports only one target");
          indicesY_.push_back(i);
          regression_ = false;
        } else if (regression_tags_.contains(head)) {
          if (!regression_ && !indicesY_.empty())
            throw std::runtime_error("currently Csv supports only one target");
          indicesY_.push_back(i);
          regression_ = true;
        } else {
          indicesX_.push_back(i);
        }
      }
    }

    void findSize() {
      size_ = std::count(
        std::istreambuf_iterator<char>(is_),
        std::istreambuf_iterator<char>(),
        '\n'
      );
    }

    Instance get() override {
      if (pos_ >= size_) throw std::out_of_range("eof");
      cacheLine();
      pos_++;

      if (cache_.size() != indicesX_.size() + indicesY_.size()) {
        print(cache_.size());
        std::string err = "unexpected number of columns on line " + std::to_string(1 + pos_);
        throw std::runtime_error(err);
      }

      engine::Tensor *tensorX, *tensorY;

      if (!indicesX_.empty()) {
        tensorX = new engine::Tensor(Float, {(unsigned) indicesX_.size()});
        auto bufferX = tensorX->malloc().as<float*>();

        for (int i = 0; i < indicesX_.size(); i++) {
          bufferX[i] = std::stof(cache_[indicesX_[i]]);
        }
      }

      if (!indicesY_.empty()) {
        tensorY = new engine::Tensor(Float, {(unsigned) indicesY_.size()});
        auto bufferY = tensorY->malloc().as<float*>();

        for (int i = 0; i < indicesY_.size(); i++) {
          bufferY[i] = std::stof(cache_[indicesY_[i]]);
        }
      }

      std::map<std::string, tensor> dict;
      if (!indicesX_.empty()) dict["x"] = engine::ref(tensorX);
      if (!indicesY_.empty()) {
        if (regression_)
          dict["y"] = engine::ref(tensorY);
        else
          dict["y"] = engine::ref(tensorY).cast(Int);
      }
      return dict;
    }

    void seek(size_t pos) override {
      seekInternal(pos);
    }

    void seekInternal(size_t pos) {
      if (pos == pos_) return;
      if (pos < pos_) {
        is_.clear();
        is_.seekg(0);
        skipLine();
        pos_ = 0;
      }
      while (pos_ < pos) {
        skipLine();
        pos_++;
      }
    }

    size_t size() const override {
      return size_;
    }

    size_t tell() const override {
      return pos_;
    }

    void cacheLine() {
      std::string lineBuffer;
      std::getline(is_, lineBuffer);
      std::stringstream lineStream(lineBuffer);
      std::string colBuffer;
      cache_.clear();
      while (std::getline(lineStream, colBuffer, ',')) {
        cache_.push_back(colBuffer);
      }
    }

    void skipLine() {
      while (is_.get() != '\n') {}
    }

  };

  return ref(new Internal(file, classification_tags, regression_tags));
}

}
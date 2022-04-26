#pragma once


#define MATCHA_DATASET_TAIL() \
  \
  Dataset internal_ = init();          \
                              \
  operator Dataset() const {  \
    return internal_;                            \
  }

#pragma once

#define MA_DATASET_TAIL()                                                \
public:                                                                   \
  Dataset::Internal* dataset_internal_ = init();                          \
  operator Dataset() {                                                  \
    return Dataset(dataset_internal_);                         \
  }                                                                       \
                                                                    \
  inline Dataset::Iterator begin() const {                                \
    return Dataset::Iterator(dataset_internal_, 0);                        \
  }                                                                 \
  Dataset::Iterator end() const {                                                     \
    return Dataset::Iterator(dataset_internal_, Dataset::Iterator::EOF_POS);                        \
  }                                                                                           \
  Batches batches(size_t batch_size);                                   \
                                                    \
  size_t size() const {                                                  \
    return dataset_internal_->size();                                                                       \
  }                                  \



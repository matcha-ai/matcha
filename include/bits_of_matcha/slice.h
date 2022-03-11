#pragma once


namespace matcha {


class Slice {
  public:
    Slice operator[](int idx);

    Slice& operator=(const Tensor& tensor);
};


}
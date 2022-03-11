#pragma once

#include <vector>
#include <initializer_list>
#include <iostream>


namespace matcha {

class Shape {
  public:
    Shape(std::initializer_list<unsigned> dims);
    Shape(const std::vector<unsigned>& dims);

    size_t rank() const;
    size_t size() const;

    unsigned operator[](int index) const;

    const unsigned* begin() const;
    const unsigned* end() const;

    bool operator==(const Shape& other) const;
    bool operator!=(const Shape& other) const;

    class Reshape {
      public:
        Reshape(const Shape& target);
        Reshape(std::initializer_list<int> target);
        Reshape(const std::vector<int>& target);

        Shape operator()(const Shape& shape) const;

      private:
        std::vector<int> target_;
        void check();
    };

  private:
    std::vector<unsigned> dims_;
};

std::ostream& operator<<(std::ostream& os, const Shape& shape);

}

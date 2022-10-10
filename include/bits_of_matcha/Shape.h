#pragma once

#include <vector>
#include <initializer_list>
#include <iostream>
#include <tuple>


namespace matcha {

class Shape final {
public:
  Shape(std::initializer_list<unsigned> dims);
  Shape(std::vector<unsigned>  dims);

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
    Reshape(std::vector<int>  target);

    Shape operator()(const Shape& shape) const;

  private:
    std::vector<int> target_;
    void check();
  };

  using Indices = std::vector<int>;

  struct Range {
    Range(bool all = true);
    Range(int idx);
    Range(int begin, int end);

    int begin, end;
  };

  class Slice {
  public:
    Slice(Slice leading, Range range);
    Slice(std::initializer_list<Range> ranges);

    std::tuple<Indices, Shape> operator()(const Shape& shape) const;
  private:
    std::vector<Range> slice_;
  };

private:
  std::vector<unsigned> dims_;
};

std::ostream& operator<<(std::ostream& os, const Shape& shape);

}

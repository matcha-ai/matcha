#pragma once

#include <string>


namespace matcha {

class Dtype {
public:

  enum {
    Float
  };

public:
  Dtype(unsigned dtype);
  Dtype(const std::string& dtype);

  std::string string() const;
  size_t size() const;

  operator unsigned() const;

private:
  unsigned dtype_;

};

enum {
  Float = Dtype::Float
};

std::ostream& operator<<(std::ostream& os, const Dtype& dtype);

}

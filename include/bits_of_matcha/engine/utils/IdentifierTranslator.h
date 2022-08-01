#pragma once

#include <map>
#include <string>


namespace matcha::engine {

template <class T>
class IdentifierTranslator {
public:
  explicit IdentifierTranslator(char alphabet_begin, char alphabet_end) {
    alphabet_begin_ = alphabet_begin;
    alphabet_end_ = alphabet_end;
  }

  explicit IdentifierTranslator()
    : IdentifierTranslator('a', 'z')
  {};

  const std::string& operator()(const T& t) {
    if (table_.contains(t)) return table_.at(t);

    size_t n = table_.size();
    size_t alphabetSize = alphabet_end_ - alphabet_begin_ + 1;
    std::string id;

    do {
      char c = n % alphabetSize + 'a';
      n /= alphabetSize;
      id += c;
    } while (n != 0);

    table_[t] = id;
    return table_.at(t);
  }

private:
  char alphabet_begin_, alphabet_end_;
  std::map<T, std::string> table_;
};

}
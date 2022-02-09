#pragma once

#include "bits_of_matcha/engine/nodeloader.h"

#include "bits_of_matcha/engine/params.h"
#include "bits_of_matcha/engine/input.h"

#include "bits_of_matcha/fn/add.h"
#include "bits_of_matcha/fn/subtract.h"
#include "bits_of_matcha/fn/multiply.h"
#include "bits_of_matcha/fn/divide.h"

#include "bits_of_matcha/fn/square.h"
#include "bits_of_matcha/fn/sqrt.h"
#include "bits_of_matcha/fn/exp.h"

#include "bits_of_matcha/fn/matmul.h"
#include "bits_of_matcha/fn/transpose.h"

#include "bits_of_matcha/fn/maxAlong.h"
#include "bits_of_matcha/fn/maxIn.h"
#include "bits_of_matcha/fn/maxBetween.h"
#include "bits_of_matcha/fn/sum.h"

#include "bits_of_matcha/fn/equal.h"

#include "bits_of_matcha/fn/normal.h"

#include <iostream>
#include <initializer_list>
#include <functional>
#include <map>


namespace matcha {
namespace engine {

class Node;

class NodeSerializer {
  public:
    NodeSerializer(std::initializer_list<std::function<const NodeLoader* ()>> loaders);
    Node* load(std::istream& is, const std::string& type, const std::vector<Tensor*>& ins) const;
    void save(std::ostream& os, Node* node) const;

    void addLoader(const NodeLoader* loader);

  private:
    std::map<std::string, const NodeLoader*> defaultRegister_;
    std::map<std::string, const NodeLoader*> extendedRegister_;

  private:
    void savePolymorphic(std::ostream& os, Node* node) const;
    void saveNonPolymorphic(std::ostream& os, Node* node) const;

};

namespace {

static NodeSerializer nodeSerializer {
  Params::loader,
  Input::loader,

  fn::Add::loader,
  fn::Subtract::loader,
  fn::Multiply::loader,
  fn::Divide::loader,

  fn::Square::loader,
  fn::Sqrt::loader,
  fn::Exp::loader,

  fn::Matmul::loader,
  fn::Transpose::loader,

  fn::MaxAlong::loader,
  fn::MaxIn::loader,
  fn::MaxBetween::loader,

  fn::Sum::loader,

  fn::Equal::loader,

  rng::Normal::loader,
};

}


};

}

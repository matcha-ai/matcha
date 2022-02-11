#pragma once

#include "bits_of_matcha/engine/tensor.h"
#include "bits_of_matcha/engine/object.h"
#include "bits_of_matcha/engine/status.h"

#include <vector>
#include <initializer_list>


namespace matcha {

class Dtype;
class Shape;

class Tensor;

namespace engine {

class NodeLoader;

class In;
class Out;


class Node : public Object {
  public:
    Node(std::initializer_list<Tensor*> ins);

  public:

    size_t ins() const;
    size_t outs() const;

    In* in(int index);
    Out* out(int index);

  public:
    void dataStatusChanged(In* in) override;
    void updateStatusChanged(In* in) override;

  protected:
    static Tensor* deref(const matcha::Tensor* tensor);
    static Tensor* deref(const matcha::Tensor& tensor);

  protected:
    std::vector<In*> ins_;
    std::vector<Out*> outs_;

    friend class Flow;
    friend class NodeSerializer;
    friend class FlowSaver;
    friend class FlowLoader;

};


}
}

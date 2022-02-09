#pragma once

#include "bits_of_matcha/engine/tensor.h"
#include "bits_of_matcha/engine/object.h"

#include <vector>
#include <initializer_list>


namespace matcha {

class Dtype;
class Shape;

class Tensor;

namespace engine {

class NodeLoader;


class Node : public Object {
  public:
    Node(std::initializer_list<Tensor*> ins);

    virtual void eval(Tensor* target) = 0;
    virtual void require() = 0;

    virtual Tensor* openIn();
    virtual Tensor* openOut();

    virtual bool openIn(Tensor* tensor);
    virtual bool openOut(Tensor* tensor);

    virtual bool closeIn(Tensor* tensor);
    virtual bool closeOut(Tensor* tensor);

    virtual bool polymorphicIns() const;
    virtual bool polymorphicOuts() const;

    virtual void onBufferChanged(int index, const device::Buffer* buffer) const;

    virtual const NodeLoader* getLoader() const = 0;
    virtual void save(std::ostream& os) const = 0;

  public:

    bool required() const;

    Tensor* out(int index) const;
    size_t outs() const;

    Tensor* in(int index) const;
    size_t ins() const;

    bool ready() const;

    void notifyBufferChanged(Tensor* tensor) const;
    void notifyReady(Tensor* tensor, bool ready) const;

  protected:
    void addIn(const Dtype& dtype, const Shape& shape);
    void addOut(const Dtype& dtype, const Shape& shape);

    void addIn(Tensor* tensor);
    void addOut(Tensor* tensor);

    void removeIn(Tensor* tensor);
    void removeOut(Tensor* tensor);

    void evalIns() const;
    void requireOuts() const;

    void unrequire() const;
    mutable bool required_;

    static Tensor* deref(const matcha::Tensor* tensor);
    static Tensor* deref(const matcha::Tensor& tensor);

  public:
    void considerPruning() override;

  protected:
    bool checkReady() const;

    mutable std::vector<Tensor*> ins_;
    mutable std::vector<Tensor*> outs_;
    mutable bool ready_;

    friend class Flow;
    friend class NodeSerializer;
    friend class FlowSaver;
    friend class FlowLoader;

};


}
}

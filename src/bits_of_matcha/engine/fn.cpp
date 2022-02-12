#include "bits_of_matcha/engine/fn.h"

#include <matcha/device>
#include <matcha/engine>
#include <algorithm>


namespace matcha {
namespace engine {

Fn::Fn(std::initializer_list<Tensor*> ins)
  : Node(ins)
  , computation_{nullptr}
{}

void Fn::wrapComputation(const std::string& name, const std::vector<In*>& ins) {
  if (computation_ != nullptr) {
    delete computation_;
  }

  std::vector<const device::Buffer*> bufferIns;
  std::transform(
    std::begin(ins), std::end(ins),
    std::back_inserter(bufferIns),
    [](auto* in) {
      return in->buffer();
    }
  );

  Debug() << "building Computation " << name;
  computation_ = Context::current()->getDevice()->createComputation(name, bufferIns);

  if (outs_.empty()) {

    for (int i = 0; i < computation_->targets(); i++) {
      auto* buffer = computation_->target(i);
      auto* out = createOut(buffer->dtype(), buffer->shape(), i);
      outs_.push_back(out);
      out->setBuffer(buffer);
    }

  } else {

    if (outs_.size() != computation_->targets()) {
      throw std::runtime_error("new computation redefines number of outs");
    }
    for (int i = 0; i < computation_->targets(); i++) {
      auto* buffer = computation_->target(i);
      if (buffer->dtype() != out(i)->dtype()  || buffer->shape() != out(i)->shape()) {
        throw std::runtime_error("new computation redefines out form");
      }
      out(i)->setBuffer(buffer);
    }

  }
}

void Fn::deduceStatus() {
  bool data = std::all_of(
    std::begin(ins_), std::end(ins_),
    [](auto* in) {
      return in->status().data;
    }
  );

  status_ = {
    .data   = data,
    .update = true,
    .ready  = false
  };
}

void Fn::dataStatusChanged(In* in) {
  bool allData = std::all_of(
    std::begin(ins_), std::end(ins_),
    [](auto* in) {
      return in->status().data;
    }
  );
  if (status_.data == allData) return;
  status_.data = allData;
  for (auto* out: outs_) out->dataStatusChanged();
}

void Fn::updateStatusChanged(In* in) {
  if (status_.update) return;
  status_.update = true;

  for (auto* out: outs_) {
    out->updateStatusChanged();
  }
}

void Fn::bufferChanged(In* in) {
  // TODO: in->id() isn't necessarily equal to its computation source index (because of Scalar0 <-> Scalar1)
//  std::cout << this << std::endl;
//  std::cout << in << std::endl;
  return;

  status_.ready = false;
  auto* buffer = in->buffer();

  if (buffer == nullptr) return;

  Debug() << "reconfiguring Computation buffer: " << this;
  computation_->source(in->id())->setSource(buffer);
}

void Fn::eval(Out* out) {
  if (!status_.data) throw std::runtime_error("data not avalable yet");
  if (!status_.update) return;
  status_.update = false;

  for (In* in: ins_) {
    in->eval();
  }

  if (!status_.ready) {
    status_.ready = true;
    computation_->prepare();
  }

  computation_->run();
}

void Fn::prune(Out* o) {
  if (referenced()) return;

  bool linked = any_of(
    std::begin(outs_), std::end(outs_),
    [](auto* out) {
      return out->linked();
    }
  );

  if (linked) return;

  for (auto* in: ins_) delete in;
  delete this;
}


}
}

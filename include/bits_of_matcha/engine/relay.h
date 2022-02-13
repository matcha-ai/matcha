#pragma once

#include "bits_of_matcha/engine/stream.h"
#include "bits_of_matcha/stream.h"


namespace matcha {
namespace engine {


class Relay : public Stream {
  public:
    Relay(std::initializer_list<Stream*> sources);

    void eval(Out* out) override;
    void close(Out* out) override;

  protected:
    Stream* source(int idx);
    const Stream* source(int idx) const;
    size_t sources() const;

  private:
    std::vector<matcha::Stream*> sources_;

};


}
}

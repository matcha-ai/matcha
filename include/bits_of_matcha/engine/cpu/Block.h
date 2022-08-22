#pragma once

#include "bits_of_matcha/engine/memory/Block.h"


namespace matcha::engine::cpu {

class Block : public engine::Block {
  public:
    explicit Block(size_t bytes);
    explicit Block(size_t bytes, void* memory);
    ~Block() override;

    void* payload() override;

  private:
    uint8_t* memory_;
};


}
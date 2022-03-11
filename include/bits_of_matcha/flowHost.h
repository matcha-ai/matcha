#pragma once


namespace matcha {

#define flow_this(idx, function, a) \
  engine::FlowContext* __flowCtxP__ = FlowHost::context(idx); \
  if (__flowCtxP__ == nullptr) {     \
    __flowCtxP__ = FlowHost::createContext(idx, [this] (auto& (a)){ return this->function(a); });                                  \
  }                                 \
  engine::FlowContext& __flowCtx__ = *__flowCtxP__;                                  \
  if (!__flowCtx__.initialized()) {                             \
    __flowCtx__.init({&(a)});                                   \
  }                                                             \
  if (__flowCtx__.built()) {                                    \
    return __flowCtx__.flow({&(a)});                            \
  }



class FlowHost {
  protected:
    engine::FlowContext* context(uint8_t idx);
    engine::FlowContext* createContext(uint8_t idx, UnaryFn fn);

  private:


};

}
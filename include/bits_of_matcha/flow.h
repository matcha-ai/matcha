#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/fn.h"
#include "device.h"

#include <variant>


namespace matcha {

class Flow;

#define FLOW1(a)          (matcha::Flow)(matcha::Flow::FromFn)(matcha::UnaryFn)[=](a) mutable -> matcha::Tensor
#define FLOW2(a, b)       (matcha::Flow)(matcha::Flow::FromFn)(matcha::BinaryFn)[=]((a), (b)) mutable -> matcha::Tensor
#define FLOW3(a, b, c)    (matcha::Flow)(matcha::Flow::FromFn)(matcha::TernaryFn)[=]((a), (b), (c)) mutable -> matcha::Tensor
#define GET_FLOW_MACRO(_1,_2,_3,NAME,...) NAME
#define flow_make(...) GET_FLOW_MACRO(__VA_ARGS__, FLOW3, FLOW2, FLOW1)(__VA_ARGS__)



class Flow {
  public:
    class FromFn {
      public:
        FromFn(const UnaryFn& fn);
        FromFn(const BinaryFn& fn);
        FromFn(const TernaryFn& fn);

      private:
        int ins_, outs_;
        std::variant<UnaryFn, BinaryFn, TernaryFn, NaryFn> fn_;
    };

    Flow(UnaryFn fn);
    explicit Flow(FromFn fn);
    static Flow init(UnaryFn fn);
    static Flow load(const std::string& file);

    Tensor operator()(const Tensor& a);

    bool built();
    void save(const std::string& file);
    void use(const Device& device);
    void use(const Device::Strategy& strategy);
    float cost();

  private:
    engine::Flow* flow_;
    UnaryFn fn_;

};


}

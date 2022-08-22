#include "bits_of_matcha/engine/op/abstract/AxiswiseFoldOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"
#include "bits_of_matcha/engine/iterations/AxiswiseFoldCtx.h"


namespace matcha::engine::ops {

struct Gather : AxiswiseFoldOp {
  explicit Gather(Tensor* a, Tensor* idxs, bool keep_dims);
  explicit Gather(Tensor* a, Tensor* idxs, int axis, bool keep_dims);
  static Reflection<Gather> reflection;

  void run() override;
};

struct GatherBack : OpBack {
  explicit GatherBack(const BackCtx& ctx);
  static Reflection<GatherBack> reflection;

  void run() override;

private:
  AxiswiseFoldCtx iter_;
};

}
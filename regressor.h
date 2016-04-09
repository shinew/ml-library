#ifndef REGRESSOR_H_
#define REGRESSOR_H_

#include "utils.h"
#include <Eigen/Core>

namespace ml {

class Regressor {
public:
  virtual void fit(const Ref<const Matrix> &X, const Ref<const Vector> &y) = 0;
  virtual Vector predict(const Ref<const Matrix> &X) const = 0;
};
}

#endif

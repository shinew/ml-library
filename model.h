#ifndef MODEL_H_
#define MODEL_H_

#include "utils.h"

namespace ml {

class Model {
public:
  virtual void fit(const Ref<const Matrix> &X, const Ref<const Vector> &y) = 0;
  virtual Vector predict(const Ref<const Matrix> &X) const = 0;
};
}

#endif

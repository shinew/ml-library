#ifndef MODEL_H_
#define MODEL_H_

#include "utils.h"

namespace ml {

template <typename Output> class Model {
public:
  virtual void fit(const Ref<const Matrix> &X, const Ref<const Output> &y) = 0;
  virtual Output predict(const Ref<const Matrix> &X) const = 0;
};
}

#endif

#ifndef NAIVE_BAYES_H_
#define NAIVE_BAYES_H_

#include "utils.h"

namespace ml {

// Multi-class Naive Bayes with Laplace smoothening.
class MultinomialNaiveBayes {
public:
  MultinomialNaiveBayes(double alpha);

  void fit(const Ref<const IMatrix> &X, const Ref<const IVector> &y);

  Matrix predict(const Ref<const IMatrix> &X) const;

private:
  bool _is_fitted;
  const double _alpha;
  Matrix _theta_xy;
  Vector _theta_y;
};
}

#endif

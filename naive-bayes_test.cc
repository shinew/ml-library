#include "naive-bayes.h"

#include "gtest/gtest.h"
#include <iostream>

namespace ml {

TEST(MultinomialNaiveBayes, Basic) {
  IMatrix X(4, 2);
  IVector y(4);
  X << 0, 1, 0, 1, 1, 0, 1, 1;
  y << 0, 0, 1, 1;

  auto nb = MultinomialNaiveBayes(1.0);
  nb.fit(X, y);

  auto X_test = IMatrix(1, 2);
  X_test << 1, 1;
  auto y_predict = nb.predict(X_test);
  std::cout << y_predict << '\n';
}
}

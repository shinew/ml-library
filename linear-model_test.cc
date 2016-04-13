#include "linear-model.h"

#include "gtest/gtest.h"
#include <vector>

namespace ml {

TEST(LinearRegression, Basic) {
  Eigen::MatrixXd X(3, 2);
  Eigen::VectorXd y(3);
  X << 1, 1, 1, 2, 1, 3;
  y << 1, 2, 3;

  auto lr = LinearRegression(0.1);
  lr.fit(X, y);

  EXPECT_NEAR(0.0, mean_squared_error(lr.predict(X), y), 1e-5);
}

TEST(LogisticRegression, Basic) {
  Eigen::MatrixXd X(4, 2);
  Eigen::VectorXd y(4);
  X << 1, 1, 1, 2, 1, 3, 1, 4;
  y << 0, 0, 1, 1;

  auto lr = LogisticRegression(0.1);
  lr.fit(X, y);

  EXPECT_NEAR(0.0, mean_squared_error(lr.predict(X), y), 1e-5);
}
}

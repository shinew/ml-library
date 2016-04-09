#include "linear-regression.h"

#include "gtest/gtest.h"

namespace ml {

namespace {

TEST(LinearRegressionError, Basic) {
  Eigen::Matrix3d X;
  Eigen::Vector3d y;
  Eigen::Vector3d theta;
  double bias = 1;
  X << 1, 1, 1, 1, 1, 1, 1, 1, 1;
  y << 1, 1, 1;
  theta << 1, 1, 1;
  EXPECT_DOUBLE_EQ(27.0, linear_regression_error(X, y, theta, bias));
}

TEST(LinearRegression, Basic) {
  Eigen::MatrixXd X (3, 2);
  Eigen::VectorXd y (3);
  X << 1, 1, 1, 2, 1, 3;
  y << 1, 2, 3;

  auto lr = LinearRegression(0.1);
  lr.fit(X, y);

  EXPECT_NEAR(0.0, mean_squared_error(lr.predict(X), y), 1e-7);
}
}
}

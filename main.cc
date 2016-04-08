#include <iostream>
#include <vector>

#include "utils.h"

int main() {
  std::vector<int> a = {1, 2, 3};
  std::vector<int> b = {0, 0, 0};
  std::cout << ml::mean_squared_error(a.begin(), a.end(), b.begin()) << '\n';
  return 0;
}

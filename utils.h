#ifndef UTILS_H_
#define UTILS_H_

#include <iterator>

namespace ml {

template <typename It> using T = typename std::iterator_traits<It>::value_type;
template <typename It>
T<It> mean_squared_error(It first1, It last1, It first2) {
  T<It> sum{};
  T<It> diff;
  for (; first1 != last1; ++first1, ++first2) {
    diff = *first1 - *first2;
    sum += diff * diff;
  }
  return sum;
}
}

#endif

// Retain the original parser and warm-up sequence.
// Allocate storage before measurement. Do not disable during Serving.
#include "activity_warm.cc"
extern "C" int activity_large_reserve() {
  records.reserve(8000000);
  return 0;
}

// Reuse the existing activity-only parser. No GPU model kernel changes.
#include "../fluidtoken_loop_split/kernel_activity.cc"
static bool prepared_once = false;
extern "C" int activity_prepare() {
  records.reserve(2000000);
  int result = activity_begin();
  if (result) return result;
  CUptiResult code = cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  if (code != CUPTI_SUCCESS) return code;
  code = cuptiActivityFlushAll(0);
  if (code != CUPTI_SUCCESS) return code;
  records.clear(); dropped = 0; invalid = 0; failure = 0;
  started = false; prepared_once = true;
  return 0;
}
extern "C" int activity_resume() {
  if (!prepared_once || started) return -10;
  records.clear(); dropped = 0; invalid = 0; failure = 0;
  CUptiResult result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  if (result == CUPTI_SUCCESS) started = true;
  return result;
}

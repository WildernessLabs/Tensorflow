#include "micro_time.h"

#include "../../../../../../Meadow/nuttx/include/nuttx/config.h"

#include <cstdint>
#include <time.h>

namespace tflite 
{

// These functions should be implemented by each target platform, and provide an
// accurate tick count along with how many ticks there are per second.
int32_t ticks_per_second()
{
    return(1e6 / CONFIG_USEC_PER_TICK);
}

// Return time in ticks.  The meaning of a tick varies per platform.
int32_t GetCurrentTimeTicks()
{
    clock_t ticks = clock();
    int32_t result;

#ifdef CONFIG_SYSTEM_TIME64
    result = ticks & 0xffffffff;
#else
    result = ticks;
#endif

    return(result);
}

}
#ifndef CHAMELEON_APPS_TOOL_LIKWID_H
#define CHAMELEON_APPS_TOOL_LIKWID_H

#ifdef LIKWID_PERFMON
#include <likwid.h>
#else /* LIKWID_PERFMON */
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_RESET(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif /* LIKWID_PERFMON */


#endif //CHAMELEON_APPS_TOOL_LIKWID_H

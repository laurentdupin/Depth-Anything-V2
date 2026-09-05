#pragma once
#if defined(__linux__) && !defined(__ANDROID__)
#include <inferbridge/linux_capture_vulkan.h>
struct dav2_context;
ibr_linux_capture_capabilities dav2_linux_capture_capabilities(dav2_context *);
void dav2_infer_linux_capture(
    dav2_context *, const inferbridge::linux_capture::LinuxDmaBufImage &,
    uint32_t, float *);
#endif

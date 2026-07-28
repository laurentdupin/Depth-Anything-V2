#include "depth_anything_v2.h"

#include <assert.h>
#include <string.h>

int main(void) {
    assert(dav2_abi_version() == DAV2_ABI_VERSION);
    assert(strlen(dav2_version_string()) != 0);
    assert(strcmp(dav2_status_string(DAV2_STATUS_OK), "ok") == 0);
    assert(
        strcmp(
            dav2_status_string(DAV2_STATUS_CANCELLED),
            "cancelled") == 0);

    dav2_image_shape shape = {0, 0};
    assert(dav2_get_network_shape(640, 480, 518, &shape) == DAV2_STATUS_OK);
    assert(shape.width == 686);
    assert(shape.height == 518);
    assert(dav2_get_network_shape(480, 640, 518, &shape) == DAV2_STATUS_OK);
    assert(shape.width == 518);
    assert(shape.height == 686);

    assert(
        dav2_get_network_shape(0, 480, 518, &shape) ==
        DAV2_STATUS_INVALID_ARGUMENT);
    assert(strlen(dav2_last_error()) != 0);
    dav2_gpu_capabilities capabilities = {0};
    capabilities.struct_size = sizeof(capabilities);
    assert(
        dav2_probe_gpu_capabilities(0, &capabilities) ==
        DAV2_STATUS_OK);
    assert(capabilities.abi_version == DAV2_ABI_VERSION);
    capabilities.struct_size = sizeof(capabilities);
    assert(
        dav2_get_gpu_capabilities(NULL, &capabilities) ==
        DAV2_STATUS_INVALID_ARGUMENT);
    dav2_d3d12_texture_submit_request texture_request = {0};
    texture_request.struct_size = sizeof(texture_request);
    texture_request.abi_version = DAV2_ABI_VERSION;
    dav2_gpu_job* job = NULL;
    assert(
        dav2_submit_d3d12_texture(
            NULL, &texture_request, &job) ==
        DAV2_STATUS_INVALID_ARGUMENT);
    return 0;
}

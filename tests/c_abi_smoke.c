#include "depth_anything_v2.h"

#include <assert.h>
#include <string.h>

int main(void) {
    assert(dav2_abi_version() == DAV2_ABI_VERSION);
    assert(strlen(dav2_version_string()) != 0);
    assert(strcmp(dav2_status_string(DAV2_STATUS_OK), "ok") == 0);

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
    return 0;
}

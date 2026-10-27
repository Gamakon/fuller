/* Futhark's runtime includes <dawn/webgpu.h> when built with -DUSE_DAWN (the
 * futures-and-wgpuInstanceWaitAny wait path). The API is the same one; the
 * header is the generated one. This file is a forwarding include only. */
#include <webgpu/webgpu.h>

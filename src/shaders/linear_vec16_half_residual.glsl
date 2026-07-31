#version 450 core

#define K_VECTORS 8
#define HALF_WEIGHT
#define COMPACT_TILE
#define FUSED_RESIDUAL
#include "linear_vec4_common.glsl"

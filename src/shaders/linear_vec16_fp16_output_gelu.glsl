#version 450 core
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define K_VECTORS 8
#define COMPACT_TILE
#define FP16_ARITHMETIC
#define FP16_OUTPUT
#define FUSED_GELU
#include "linear_vec4_common.glsl"

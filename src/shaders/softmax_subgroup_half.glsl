#version 450 core
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#define HALF_VALUES
#include "softmax_subgroup_common.glsl"

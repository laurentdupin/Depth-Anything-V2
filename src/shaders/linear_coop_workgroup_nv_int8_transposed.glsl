#version 450 core
#pragma use_vulkan_memory_model
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_NV_cooperative_matrix2 : require

layout(local_size_x=128,local_size_y=1,local_size_z=1) in;
layout(set=0,binding=0,std430) writeonly buffer O{float d[];}o;
layout(set=0,binding=1,std430) readonly buffer I{int8_t d[];}i;
layout(set=0,binding=2,std430) readonly buffer W{int8_t d[];}w;
layout(set=0,binding=3,std430) readonly buffer IS{float d[];}is;
layout(set=0,binding=4,std430) readonly buffer WS{float d[];}ws;
layout(set=0,binding=5,std430) readonly buffer B{float d[];}b;
layout(push_constant) uniform P{uint rows;uint ic;uint oc;}p;
const uint TILE_M=32,TILE_N=32,TILE_K=32;

void main(){
 uint row0=gl_WorkGroupID.y*TILE_M,col0=gl_WorkGroupID.x*TILE_N;
 coopmat<int,gl_ScopeWorkgroup,TILE_M,TILE_N,gl_MatrixUseAccumulator> sum=
  coopmat<int,gl_ScopeWorkgroup,TILE_M,TILE_N,gl_MatrixUseAccumulator>(0);
 tensorLayoutNV<2,gl_CooperativeMatrixClampModeConstantNV> la=createTensorLayoutNV(2,gl_CooperativeMatrixClampModeConstantNV);
 tensorLayoutNV<2,gl_CooperativeMatrixClampModeConstantNV> lw=createTensorLayoutNV(2,gl_CooperativeMatrixClampModeConstantNV);
 tensorLayoutNV<2,gl_CooperativeMatrixClampModeConstantNV> lo=createTensorLayoutNV(2,gl_CooperativeMatrixClampModeConstantNV);
 tensorLayoutNV<2,gl_CooperativeMatrixClampModeRepeatNV> lis=createTensorLayoutNV(2,gl_CooperativeMatrixClampModeRepeatNV);
 tensorLayoutNV<2,gl_CooperativeMatrixClampModeRepeatNV> lws=createTensorLayoutNV(2,gl_CooperativeMatrixClampModeRepeatNV);
 la=setTensorLayoutDimensionNV(la,p.rows,p.ic);
 lw=setTensorLayoutDimensionNV(lw,p.ic,p.oc);
 lo=setTensorLayoutDimensionNV(lo,p.rows,p.oc);
 lis=setTensorLayoutDimensionNV(lis,p.rows,1);
 lws=setTensorLayoutDimensionNV(lws,1,p.oc);
 for(uint k0=0;k0<p.ic;k0+=TILE_K){
  coopmat<int8_t,gl_ScopeWorkgroup,TILE_M,TILE_K,gl_MatrixUseA> ma;
  coopmat<int8_t,gl_ScopeWorkgroup,TILE_K,TILE_N,gl_MatrixUseB> mw;
  coopMatLoadTensorNV(ma,i.d,0,sliceTensorLayoutNV(la,int(row0),TILE_M,int(k0),TILE_K));
  coopMatLoadTensorNV(mw,w.d,0,sliceTensorLayoutNV(lw,int(k0),TILE_K,int(col0),TILE_N));
  sum=coopMatMulAdd(ma,mw,sum);
 }
 coopmat<float,gl_ScopeWorkgroup,TILE_M,TILE_N,gl_MatrixUseAccumulator> result=
  coopmat<float,gl_ScopeWorkgroup,TILE_M,TILE_N,gl_MatrixUseAccumulator>(sum);
 coopmat<float,gl_ScopeWorkgroup,TILE_M,TILE_N,gl_MatrixUseAccumulator> input_scale;
 coopmat<float,gl_ScopeWorkgroup,TILE_M,TILE_N,gl_MatrixUseAccumulator> weight_scale;
 coopmat<float,gl_ScopeWorkgroup,TILE_M,TILE_N,gl_MatrixUseAccumulator> bias;
 coopMatLoadTensorNV(input_scale,is.d,0,sliceTensorLayoutNV(lis,int(row0),TILE_M,0,TILE_N));
 coopMatLoadTensorNV(weight_scale,ws.d,0,sliceTensorLayoutNV(lws,0,TILE_M,int(col0),TILE_N));
 coopMatLoadTensorNV(bias,b.d,0,sliceTensorLayoutNV(lws,0,TILE_M,int(col0),TILE_N));
 result=result*input_scale*weight_scale+bias;
 coopMatStoreTensorNV(result,o.d,0,sliceTensorLayoutNV(lo,int(row0),TILE_M,int(col0),TILE_N));
}

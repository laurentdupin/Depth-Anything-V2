#version 450 core
#pragma use_vulkan_memory_model
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_NV_cooperative_matrix2 : require

layout(local_size_x=128,local_size_y=1,local_size_z=1) in;
layout(set=0,binding=0,std430) writeonly buffer O{float16_t d[];}o;
layout(set=0,binding=1,std430) readonly buffer I{float16_t d[];}i;
layout(set=0,binding=2,std430) readonly buffer W{float16_t d[];}w;
layout(set=0,binding=3,std430) readonly buffer B{float d[];}b;
layout(push_constant) uniform P{uint rows;uint ic;uint oc;}p;
const uint TILE_M=32,TILE_N=32,TILE_K=16;

float gelu_element(const in uint row,const in uint col,const in float v){
 float x=v*0.7071067811865475,m=abs(x),t=1.0/(1.0+0.5*m);
 float z=-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+t*(-0.82215223+t*0.17087277))))))));
 float e=t*exp(-m*m+z),er=x>=0?1.0-e:e-1.0;
 return 0.5*v*(1.0+er);
}

void main(){
 uint row0=gl_WorkGroupID.y*TILE_M,col0=gl_WorkGroupID.x*TILE_N;
 coopmat<float,gl_ScopeWorkgroup,TILE_M,TILE_N,gl_MatrixUseAccumulator> sum=
  coopmat<float,gl_ScopeWorkgroup,TILE_M,TILE_N,gl_MatrixUseAccumulator>(0.0);
 tensorLayoutNV<2,gl_CooperativeMatrixClampModeConstantNV> la=createTensorLayoutNV(2,gl_CooperativeMatrixClampModeConstantNV);
 tensorLayoutNV<2,gl_CooperativeMatrixClampModeConstantNV> lw=createTensorLayoutNV(2,gl_CooperativeMatrixClampModeConstantNV);
 tensorLayoutNV<2,gl_CooperativeMatrixClampModeConstantNV> lo=createTensorLayoutNV(2,gl_CooperativeMatrixClampModeConstantNV);
 tensorLayoutNV<2,gl_CooperativeMatrixClampModeRepeatNV> lb=createTensorLayoutNV(2,gl_CooperativeMatrixClampModeRepeatNV);
 la=setTensorLayoutDimensionNV(la,p.rows,p.ic);
 lw=setTensorLayoutDimensionNV(lw,p.ic,p.oc);
 lo=setTensorLayoutDimensionNV(lo,p.rows,p.oc);
 lb=setTensorLayoutDimensionNV(lb,1,p.oc);
 for(uint k0=0;k0<p.ic;k0+=TILE_K){
  coopmat<float16_t,gl_ScopeWorkgroup,TILE_M,TILE_K,gl_MatrixUseA> ma;
  coopmat<float16_t,gl_ScopeWorkgroup,TILE_K,TILE_N,gl_MatrixUseB> mw;
  coopMatLoadTensorNV(ma,i.d,0,sliceTensorLayoutNV(la,int(row0),TILE_M,int(k0),TILE_K));
  coopMatLoadTensorNV(mw,w.d,0,sliceTensorLayoutNV(lw,int(k0),TILE_K,int(col0),TILE_N));
  sum=coopMatMulAdd(ma,mw,sum);
 }
 coopmat<float,gl_ScopeWorkgroup,TILE_M,TILE_N,gl_MatrixUseAccumulator> bias;
 coopMatLoadTensorNV(bias,b.d,0,sliceTensorLayoutNV(lb,0,TILE_M,int(col0),TILE_N));
 sum+=bias;
 coopmat<float,gl_ScopeWorkgroup,TILE_M,TILE_N,gl_MatrixUseAccumulator> activated;
 coopMatPerElementNV(activated,sum,gelu_element);
 coopmat<float16_t,gl_ScopeWorkgroup,TILE_M,TILE_N,gl_MatrixUseAccumulator> packed=coopmat<float16_t,gl_ScopeWorkgroup,TILE_M,TILE_N,gl_MatrixUseAccumulator>(activated);
 coopMatStoreTensorNV(packed,o.d,0,sliceTensorLayoutNV(lo,int(row0),TILE_M,int(col0),TILE_N));
}

#version 450 core
#pragma use_vulkan_memory_model
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_NV_cooperative_matrix2 : require
layout(local_size_x=128,local_size_y=1,local_size_z=1) in;
layout(set=0,binding=0,std430) writeonly buffer O{float d[];}o;
layout(set=0,binding=1,std430) readonly buffer I{float16_t d[];}i;
layout(set=0,binding=2,std430) readonly buffer W{float16_t d[];}w;
layout(set=0,binding=3,std430) readonly buffer B{float d[];}b;
layout(push_constant) uniform P{uint rows;uint ic;uint oc;}p;
const uint TM=32,TN=32,TK=16;
void main(){
 uint row0=gl_WorkGroupID.y*TM,col0=gl_WorkGroupID.x*TN;
 coopmat<float,gl_ScopeWorkgroup,TM,TN,gl_MatrixUseAccumulator> sum=
  coopmat<float,gl_ScopeWorkgroup,TM,TN,gl_MatrixUseAccumulator>(0.0);
 tensorLayoutNV<2,gl_CooperativeMatrixClampModeConstantNV> li=createTensorLayoutNV(2,gl_CooperativeMatrixClampModeConstantNV);
 tensorLayoutNV<2,gl_CooperativeMatrixClampModeConstantNV> lw=createTensorLayoutNV(2,gl_CooperativeMatrixClampModeConstantNV);
 tensorLayoutNV<2,gl_CooperativeMatrixClampModeConstantNV> lo=createTensorLayoutNV(2,gl_CooperativeMatrixClampModeConstantNV);
 tensorLayoutNV<2,gl_CooperativeMatrixClampModeRepeatNV> lb=createTensorLayoutNV(2,gl_CooperativeMatrixClampModeRepeatNV);
 li=setTensorLayoutDimensionNV(li,p.ic,p.rows);
 lw=setTensorLayoutDimensionNV(lw,p.oc,p.ic);
 lo=setTensorLayoutDimensionNV(lo,p.oc,p.rows);
 lb=setTensorLayoutDimensionNV(lb,1,p.oc);
 tensorViewNV<2,false,1,0> tr=createTensorViewNV(2,false,1,0);
 for(uint k0=0;k0<p.ic;k0+=TK){
  coopmat<float16_t,gl_ScopeWorkgroup,TM,TK,gl_MatrixUseA> a;
  coopmat<float16_t,gl_ScopeWorkgroup,TK,TN,gl_MatrixUseB> weight;
  coopMatLoadTensorNV(a,i.d,0,sliceTensorLayoutNV(li,int(k0),TK,int(row0),TM),tr);
  coopMatLoadTensorNV(weight,w.d,0,sliceTensorLayoutNV(lw,int(col0),TN,int(k0),TK),tr);
  sum=coopMatMulAdd(a,weight,sum);
 }
 coopmat<float,gl_ScopeWorkgroup,TM,TN,gl_MatrixUseAccumulator> bias;
 coopMatLoadTensorNV(bias,b.d,0,sliceTensorLayoutNV(lb,0,TM,int(col0),TN));
 sum+=bias;
 coopMatStoreTensorNV(sum,o.d,0,sliceTensorLayoutNV(lo,int(col0),TN,int(row0),TM),tr);
}

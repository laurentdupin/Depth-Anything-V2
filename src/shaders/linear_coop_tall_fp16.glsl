#version 450 core
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require

layout(local_size_x=128,local_size_y=1,local_size_z=1) in;
layout(set=0,binding=0,std430) writeonly buffer O{float d[];}o;
layout(set=0,binding=1,std430) readonly buffer I{float16_t d[];}i;
layout(set=0,binding=2,std430) readonly buffer W{float16_t d[];}w;
layout(set=0,binding=3,std430) readonly buffer B{float d[];}b;
layout(push_constant) uniform P{uint rows;uint ic;uint oc;}p;
shared float16_t at[1024];shared float16_t bt[256];shared float ct[1024];
void main(){uint lid=gl_LocalInvocationID.x,sg=gl_SubgroupID;
 uint r0=gl_WorkGroupID.y*64,c0=gl_WorkGroupID.x*16;
 coopmat<float,gl_ScopeSubgroup,16,16,gl_MatrixUseAccumulator> sum=
  coopmat<float,gl_ScopeSubgroup,16,16,gl_MatrixUseAccumulator>(0.0);
 for(uint kb=0;kb<p.ic;kb+=16){
  for(uint n=lid;n<1024;n+=128){uint r=n/16,k=n%16,rr=r0+r;
   at[n]=rr<p.rows&&kb+k<p.ic?i.d[rr*p.ic+kb+k]:float16_t(0);}
  for(uint n=lid;n<256;n+=128){uint k=n/16,c=n%16,cc=c0+c;
   bt[n]=cc<p.oc&&kb+k<p.ic?w.d[cc*p.ic+kb+k]:float16_t(0);}
  barrier();
  coopmat<float16_t,gl_ScopeSubgroup,16,16,gl_MatrixUseA> ma;
  coopmat<float16_t,gl_ScopeSubgroup,16,16,gl_MatrixUseB> mb;
  coopMatLoad(ma,at,sg*256,16,gl_CooperativeMatrixLayoutRowMajor);
  coopMatLoad(mb,bt,0,16,gl_CooperativeMatrixLayoutRowMajor);
  sum=coopMatMulAdd(ma,mb,sum);barrier();
 }
 coopMatStore(sum,ct,sg*256,16,gl_CooperativeMatrixLayoutRowMajor);barrier();
 for(uint n=lid;n<1024;n+=128){uint r=n/16,c=n%16,rr=r0+r,cc=c0+c;
  if(rr<p.rows&&cc<p.oc)o.d[rr*p.oc+cc]=ct[n]+b.d[cc];}
}

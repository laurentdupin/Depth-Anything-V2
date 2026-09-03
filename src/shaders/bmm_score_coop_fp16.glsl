#version 450 core
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(set=0,binding=0,std430) writeonly buffer O{uint d[];}o;
layout(set=0,binding=1,std430) readonly buffer Q{float d[];}q;
layout(push_constant) uniform P{uint tokens;uint heads;}p;
shared float16_t a[256]; shared float16_t b[256]; shared float c[256];
void main(){
 uint lane=gl_LocalInvocationID.x,row0=gl_WorkGroupID.y*16,col0=gl_WorkGroupID.x*16;
 uint head=gl_WorkGroupID.z,embedding=p.heads*64;
 coopmat<float,gl_ScopeSubgroup,16,16,gl_MatrixUseAccumulator> sum=
  coopmat<float,gl_ScopeSubgroup,16,16,gl_MatrixUseAccumulator>(0.0);
 for(uint kb=0;kb<64;kb+=16){
  for(uint n=lane;n<256;n+=32){uint r=n/16,k=n%16,rt=row0+r,ct=col0+r;
   a[n]=rt<p.tokens?float16_t(q.d[rt*embedding*3+head*64+kb+k]*0.125):float16_t(0);
   b[k*16+r]=ct<p.tokens?float16_t(q.d[ct*embedding*3+embedding+head*64+kb+k]):float16_t(0);}
  barrier();
  coopmat<float16_t,gl_ScopeSubgroup,16,16,gl_MatrixUseA> ma;
  coopmat<float16_t,gl_ScopeSubgroup,16,16,gl_MatrixUseB> mb;
  coopMatLoad(ma,a,0,16,gl_CooperativeMatrixLayoutRowMajor);
  coopMatLoad(mb,b,0,16,gl_CooperativeMatrixLayoutRowMajor);
  sum=coopMatMulAdd(ma,mb,sum); barrier();
 }
 coopMatStore(sum,c,0,16,gl_CooperativeMatrixLayoutRowMajor); barrier();
 uint words=(p.tokens+1)>>1;
 for(uint n=lane;n<128;n+=32){uint r=n/8,pair=n%8,rt=row0+r,ct=col0+pair*2;
  if(rt<p.tokens&&ct<p.tokens){float hi=ct+1<p.tokens?c[r*16+pair*2+1]:0;
   o.d[(head*p.tokens+rt)*words+ct/2]=packHalf2x16(vec2(c[r*16+pair*2],hi));}}
}

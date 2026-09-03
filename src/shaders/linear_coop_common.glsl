layout(local_size_x=32,local_size_y=1,local_size_z=1) in;
#if defined(HALF_OUTPUT)
layout(set=0,binding=0,std430) writeonly buffer O{float16_t d[];}o;
#else
layout(set=0,binding=0,std430) writeonly buffer O{float d[];}o;
#endif
layout(set=0,binding=1,std430) readonly buffer I{float16_t d[];}i;
layout(set=0,binding=2,std430) readonly buffer W{float16_t d[];}w;
layout(set=0,binding=3,std430) readonly buffer B{float d[];}b;
layout(push_constant) uniform P{uint rows;uint ic;uint oc;}p;
shared float16_t at[256];shared float16_t bt[256];shared float ct[256];
#if defined(FUSED_GELU)
float exact_gelu(float v){float x=v*0.7071067811865475,m=abs(x),t=1.0/(1.0+0.5*m);
 float z=-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+t*(-0.82215223+t*0.17087277))))))));
 float e=t*exp(-m*m+z),er=x>=0?1.0-e:e-1.0;return 0.5*v*(1.0+er);}
#endif
void main(){uint lane=gl_LocalInvocationID.x,r0=gl_WorkGroupID.y*16,c0=gl_WorkGroupID.x*16;
 coopmat<float,gl_ScopeSubgroup,16,16,gl_MatrixUseAccumulator> sum=
  coopmat<float,gl_ScopeSubgroup,16,16,gl_MatrixUseAccumulator>(0.0);
 for(uint kb=0;kb<p.ic;kb+=16){for(uint n=lane;n<256;n+=32){uint r=n/16,k=n%16,rr=r0+r,cc=c0+r;
   at[n]=rr<p.rows&&kb+k<p.ic?i.d[rr*p.ic+kb+k]:float16_t(0);
   bt[k*16+r]=cc<p.oc&&kb+k<p.ic?w.d[cc*p.ic+kb+k]:float16_t(0);}
  barrier();coopmat<float16_t,gl_ScopeSubgroup,16,16,gl_MatrixUseA> ma;
  coopmat<float16_t,gl_ScopeSubgroup,16,16,gl_MatrixUseB> mb;
  coopMatLoad(ma,at,0,16,gl_CooperativeMatrixLayoutRowMajor);
  coopMatLoad(mb,bt,0,16,gl_CooperativeMatrixLayoutRowMajor);
  sum=coopMatMulAdd(ma,mb,sum);barrier();}
 coopMatStore(sum,ct,0,16,gl_CooperativeMatrixLayoutRowMajor);barrier();
 for(uint n=lane;n<256;n+=32){uint r=n/16,c=n%16,rr=r0+r,cc=c0+c;if(rr<p.rows&&cc<p.oc){float v=ct[n]+b.d[cc];
#if defined(FUSED_GELU)
  v=exact_gelu(v);
#endif
  o.d[rr*p.oc+cc]=
#if defined(HALF_OUTPUT)
   float16_t(v);
#else
   v;
#endif
 }}
}

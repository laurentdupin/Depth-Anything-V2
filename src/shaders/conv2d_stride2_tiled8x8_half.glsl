#version 450 core
layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
layout(set=0,binding=0,std430) writeonly buffer O{float d[];}o;
layout(set=0,binding=1,std430) readonly buffer I{float d[];}i;
layout(set=0,binding=2,std430) readonly buffer W{uint d[];}w;
layout(set=0,binding=3,std430) readonly buffer B{float d[];}b;
layout(push_constant) uniform P{uint iw;uint ih;uint ic;uint ow;uint oh;uint oc;
 uint kernel;uint stride;int padding;uint has_bias;}p;
shared float st[8*17*17];shared float kt[8*8*9];
float rw(uint n){vec2 v=unpackHalf2x16(w.d[n>>1]);return (n&1)==0?v.x:v.y;}
void main(){
 uint x=gl_GlobalInvocationID.x,y=gl_GlobalInvocationID.y,cb=gl_WorkGroupID.z*8;
 bool valid=x<p.ow&&y<p.oh&&cb<p.oc;float sums[8];for(uint n=0;n<8;++n)sums[n]=0;
 uint lane=gl_LocalInvocationID.y*8+gl_LocalInvocationID.x;
 int ox=int(gl_WorkGroupID.x*16)-p.padding,oy=int(gl_WorkGroupID.y*16)-p.padding;
 for(uint ib=0;ib<p.ic;ib+=8){
  for(uint n=lane;n<8*17*17;n+=64){uint ci=n/(17*17),si=n%(17*17),ch=ib+ci;
   int px=ox+int(si%17),py=oy+int(si/17);
   st[n]=ch<p.ic&&px>=0&&px<int(p.iw)&&py>=0&&py<int(p.ih)
    ?i.d[(ch*p.ih+uint(py))*p.iw+uint(px)]:0;}
  for(uint n=lane;n<8*8*9;n+=64){uint ci=n/(8*9),r=n%(8*9),co=r/9,k=r%9;
   uint ich=ib+ci,och=cb+co;kt[n]=ich<p.ic&&och<p.oc?rw((och*p.ic+ich)*9+k):0;}
  barrier();
  if(valid)for(uint ci=0;ci<8&&ib+ci<p.ic;++ci)
   for(uint ky=0;ky<3;++ky)for(uint kx=0;kx<3;++kx){
    float v=st[ci*17*17+(gl_LocalInvocationID.y*2+ky)*17+gl_LocalInvocationID.x*2+kx];
    uint k=ky*3+kx;for(uint co=0;co<8;++co)sums[co]+=v*kt[ci*8*9+co*9+k];}
  barrier();
 }
 if(!valid)return;for(uint co=0;co<8;++co){uint ch=cb+co;if(ch<p.oc)
  o.d[(ch*p.oh+y)*p.ow+x]=sums[co]+(p.has_bias!=0?b.d[ch]:0);}
}

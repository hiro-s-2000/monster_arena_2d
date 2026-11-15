// HD-2D post-processing on WebGL1. Inputs are three art-resolution canvases:
//   world (lit terrain + sprites, sampled NEAREST), fx (effects, NEAREST, alpha ordered-dithered and
//   posterized here so smooth canvas gradients become retro dithered pixels), light (smooth multiplier, LINEAR).
// Passes: compose → bright-pass → 2× separable blur (bloom) and a blurred copy for tilt-shift depth of field
// → final grade (tilt-shift, bloom, fog, god rays, vignette, flash, grain).
"use strict";

const Post = (() => {
  let gl = null, ok = false, quad, P = {}, T = {}, F = {};
  let W = 0, H = 0, QW = 0, QH = 0;

  const VS = "attribute vec2 p;varying vec2 v;void main(){v=p*0.5+0.5;gl_Position=vec4(p,0.,1.);}";
  const COMMON = "precision mediump float;varying vec2 v;";
  const FS = {
    // [ユーザー指定 2026-11-15] 「バトルステージを、今の特徴を維持したまま、かなりリアルに近い HD-2D のフィールドに加工」。
    // world（粗いドットのステージ）を画面の解像度で加工する（uHd.x > 0 のとき）。モンスター（uSpr）と技（uFx）はドットのまま：
    //  1) なめらかな拡大：まわり 3×3 のドットを、色が近いものだけ混ぜる（網点の濃淡はなめらかな階調に、白線や観客などの強い輪郭は残る）
    //  2) 起伏の陰影：明るさを高さとみなして左上からの光を当てる（石の継ぎ目・土のひび・芝の濃淡が凹凸に見える）
    //  3) 素材の細かい肌：ステージに固定した細かい凹凸（芝・土・石・氷…ごとに向きと強さを変える）を高さと色に足す
    //  4) 照り返し：光の向きに傾いた面だけ光らせる（氷・水・石は強く、芝や土は弱く）
    // uHd = (混ぜる色の近さ, 起伏の強さ, 照り返しの強さ, 肌の強さ)、uMat = (肌の細かさ x, y, 照り返しの鋭さ)、uOrg = バッファ左上のステージ座標。
    compose: `
      #ifdef GL_FRAGMENT_PRECISION_HIGH
      precision highp float;
      #else
      precision mediump float;
      #endif
      varying vec2 v;
      uniform sampler2D uWorld,uFx,uLight,uSpr;uniform vec2 uLow,uRes,uFrac,uSprLow,uSprOff,uOrg;uniform float uS,uZoom,uPad,uTime,uSprK;
      uniform vec4 uHd;uniform vec3 uMat;
      uniform vec3 uSunCol,uShCol;uniform float uSun;uniform vec4 uFld;
      float b2(vec2 a){a=floor(a);return fract(dot(a,vec2(0.5,a.y*0.75)));}
      float bayer(vec2 a){return b2(0.5*a)*0.25+b2(a);}
      float hh(vec2 q){return fract(sin(dot(q,vec2(127.1,311.7)))*43758.5453);}
      float vn(vec2 q){vec2 i=floor(q),f=fract(q);f=f*f*(3.0-2.0*f);return mix(mix(hh(i),hh(i+vec2(1.0,0.0)),f.x),mix(hh(i+vec2(0.0,1.0)),hh(i+vec2(1.0,1.0)),f.x),f.y);}
      float lum(vec3 c){return dot(c,vec3(0.299,0.587,0.114));}
      vec3 tx(vec2 c){return texture2D(uWorld,(c+0.5)/uLow).rgb;}
      float micro(vec2 q,vec3 k){return (vn(q*0.5)-0.5)*0.5*k.x+(vn(q*1.1+17.0)-0.5)*0.3*k.y+(vn(q*2.3+41.0)-0.5)*0.2*k.z+0.5;}
      void main(){
        vec2 sp=vec2(gl_FragCoord.x,uRes.y-gl_FragCoord.y);
        sp=(sp-uRes*0.5)/uZoom+uRes*0.5*uPad;
        vec2 p=sp/uS+uFrac; vec2 cell=floor(p);
        vec3 w=tx(cell);
        if(uHd.x>0.0){
          vec3 acc=vec3(0.0);float ws=0.0;
          for(int j=-1;j<=1;j++){for(int i=-1;i<=1;i++){
            vec2 o=vec2(float(i),float(j));vec3 s=tx(cell+o);
            vec2 d=cell+o+0.5-p;vec3 dc=s-w;
            float k=exp(-dot(d,d)*0.95)*exp(-dot(dc,dc)/(uHd.x*uHd.x));
            acc+=s*k;ws+=k;}}
          vec3 alb=acc/ws;
          float e=0.85;
          float gx=lum(texture2D(uWorld,(p+vec2(e,0.0))/uLow).rgb)-lum(texture2D(uWorld,(p-vec2(e,0.0))/uLow).rgb);
          float gy=lum(texture2D(uWorld,(p+vec2(0.0,e))/uLow).rgb)-lum(texture2D(uWorld,(p-vec2(0.0,e))/uLow).rgb);
          gx=clamp(gx,-0.22,0.22);gy=clamp(gy,-0.22,0.22);
          // 肌の細かさは画面の画素より細かくしない（1 周期が 2.5 画素を切る成分は消す）。細かすぎるとカメラが動くたびに全面がちらつく。
          float ppc=uS*uZoom/max(uMat.x,uMat.y);
          vec3 fine=vec3(smoothstep(2.5,5.0,ppc/0.5),smoothstep(2.5,5.0,ppc/1.1),smoothstep(2.5,5.0,ppc/2.3));
          vec2 q=(p+uOrg)*uMat.xy;
          float m0=micro(q,fine),mx=micro(q+vec2(0.35,0.0),fine),my=micro(q+vec2(0.0,0.35),fine);
          vec3 n=normalize(vec3(-(gx*uHd.y*3.2+(mx-m0)*uHd.w*2.4),-(gy*uHd.y*3.2+(my-m0)*uHd.w*2.4),1.0));
          vec3 Ld=normalize(vec3(-0.45,-0.6,0.66));
          float dif=dot(n,Ld)/Ld.z;
          float la=lum(alb);
          alb*=1.0+(m0-0.5)*uHd.w*0.3;
          float spc=pow(max(dot(n,normalize(Ld+vec3(0.0,0.0,1.0))),0.0),uMat.z)*uHd.z;
          float keep=smoothstep(0.03,0.12,la);
          w=mix(w,alb*clamp(mix(1.0,dif,0.85),0.5,1.6)+spc*(0.35+la),keep);
        }
        // [ユーザー指定 2026-11-16] HD-2D の「木漏れ日」: 地面に、斜め（左上→右下）に伸びた日なたと日かげの帯を落とす。ステージに固定し、雲のようにゆっくり流れる。
        // 日なたは暖色で明るく、日かげは寒色で暗く（uSunCol / uShCol）。モンスターには掛けない（体が縞にならないように）。
        if(uSun>0.0){
          // 大きくやわらかい日なた・日かげ（雲の影のような広い面）。細い縞にしない。フィールドの中だけ（観客席や壁には掛けない）。
          vec2 wq=p+uOrg; vec2 r2=vec2(wq.x*0.82+wq.y*0.57,(wq.y*0.82-wq.x*0.57)*1.5);
          float cl=vn(r2*0.0052+vec2(uTime*0.006,uTime*0.002))*0.72+vn(r2*0.013+vec2(-uTime*0.009,9.0))*0.28;
          float lit=smoothstep(0.33,0.67,cl);
          vec2 fd=min(wq-uFld.xy,uFld.xy+uFld.zw-wq);
          float inF=uFld.z>0.0?smoothstep(-2.0,8.0,min(fd.x,fd.y)):1.0;
          w*=mix(vec3(1.0),mix(uShCol,uSunCol,lit),uSun*inF);
        }
        // Monster layer: its own window (uSprOff, art px) at uSprK texels per art px, premultiplied alpha.
        // Finer than the screen → plain linear (keeps every sprite pixel when shrinking); coarser → sharp-bilinear (hard pixel edges).
        vec2 q=(p-uSprOff)*uSprK; float r=uSprK/(uS*uZoom);
        vec2 uq=r>=1.0?q:floor(q)+0.5+clamp((fract(q)-0.5)/r,-0.5,0.5);
        vec4 spr=texture2D(uSpr,uq/uSprLow);
        if(q.x<0.0||q.y<0.0||q.x>uSprLow.x||q.y>uSprLow.y) spr=vec4(0.0);
        vec3 L=texture2D(uLight,p/uLow).rgb*2.0;
        vec4 f=texture2D(uFx,(cell+0.5)/uLow);
        float th=bayer(cell)+0.03;
        float fa=f.a<0.02?0.0:step(th,f.a*1.12);
        vec3 fc=floor(f.rgb*6.0+0.5)/6.0;
        vec3 Ls=min(L,1.0)+max(L-1.0,0.0)*0.3; // モンスターだけ、強い光を圧縮（spr3 の SPR_OVER と同じ）
        vec3 base=w*(1.0-spr.a)*L+spr.rgb*Ls;
        vec3 col=mix(base,fc,fa);
        float em=fa*max(fc.r,max(fc.g,fc.b));
        gl_FragColor=vec4(col,em);
      }`,
    // ---- 3D（Scene3D）用: 地面を z = 0 の面として遠近で描く。HD 加工は compose と同じ式 ----
    ground3: `
      #ifdef GL_FRAGMENT_PRECISION_HIGH
      precision highp float;
      #else
      precision mediump float;
      #endif
      varying vec2 vP;
      uniform sampler2D uWorld,uLight;uniform vec2 uLow;uniform float uTime,uF;
      uniform vec4 uHd;uniform vec3 uMat;
      uniform vec3 uSunCol,uShCol;uniform float uSun;uniform vec4 uFld;
      uniform vec4 uClip;uniform float uClipR;uniform vec3 uFog,uFogCol;
      float hh(vec2 q){return fract(sin(dot(q,vec2(127.1,311.7)))*43758.5453);}
      float vn(vec2 q){vec2 i=floor(q),f=fract(q);f=f*f*(3.0-2.0*f);return mix(mix(hh(i),hh(i+vec2(1.0,0.0)),f.x),mix(hh(i+vec2(0.0,1.0)),hh(i+vec2(1.0,1.0)),f.x),f.y);}
      float lum(vec3 c){return dot(c,vec3(0.299,0.587,0.114));}
      vec3 tx(vec2 c){return texture2D(uWorld,(c+0.5)/uLow).rgb;}
      float micro(vec2 q,vec3 k){return (vn(q*0.5)-0.5)*0.5*k.x+(vn(q*1.1+17.0)-0.5)*0.3*k.y+(vn(q*2.3+41.0)-0.5)*0.2*k.z+0.5;}
      void main(){
        vec2 p=vP; vec2 cell=floor(p);
        if(uClip.z>0.0){vec2 d=abs(p-(uClip.xy+uClip.zw*0.5))-uClip.zw*0.5+uClipR;if(length(max(d,0.0))+min(max(d.x,d.y),0.0)-uClipR>0.0)discard;}
        float ppa=uF*gl_FragCoord.w; // この画素での 1 art px の画素数
        vec3 w=tx(cell);
        if(uHd.x>0.0){
          vec3 acc=vec3(0.0);float ws=0.0;
          for(int j=-1;j<=1;j++){for(int i=-1;i<=1;i++){
            vec2 o=vec2(float(i),float(j));vec3 s=tx(cell+o);
            vec2 d=cell+o+0.5-p;vec3 dc=s-w;
            float k=exp(-dot(d,d)*0.95)*exp(-dot(dc,dc)/(uHd.x*uHd.x));
            acc+=s*k;ws+=k;}}
          vec3 alb=acc/ws;
          float e=0.85;
          float gx=lum(texture2D(uWorld,(p+vec2(e,0.0))/uLow).rgb)-lum(texture2D(uWorld,(p-vec2(e,0.0))/uLow).rgb);
          float gy=lum(texture2D(uWorld,(p+vec2(0.0,e))/uLow).rgb)-lum(texture2D(uWorld,(p-vec2(0.0,e))/uLow).rgb);
          gx=clamp(gx,-0.22,0.22);gy=clamp(gy,-0.22,0.22);
          float ppc=ppa/max(uMat.x,uMat.y);
          vec3 fine=vec3(smoothstep(2.5,5.0,ppc/0.5),smoothstep(2.5,5.0,ppc/1.1),smoothstep(2.5,5.0,ppc/2.3));
          vec2 q=p*uMat.xy;
          float m0=micro(q,fine),mx=micro(q+vec2(0.35,0.0),fine),my=micro(q+vec2(0.0,0.35),fine);
          vec3 n=normalize(vec3(-(gx*uHd.y*3.2+(mx-m0)*uHd.w*2.4),-(gy*uHd.y*3.2+(my-m0)*uHd.w*2.4),1.0));
          vec3 Ld=normalize(vec3(-0.45,-0.6,0.66));
          float dif=dot(n,Ld)/Ld.z;
          float la=lum(alb);
          alb*=1.0+(m0-0.5)*uHd.w*0.3;
          float spc=pow(max(dot(n,normalize(Ld+vec3(0.0,0.0,1.0))),0.0),uMat.z)*uHd.z;
          float keep=smoothstep(0.03,0.12,la);
          w=mix(w,alb*clamp(mix(1.0,dif,0.85),0.5,1.6)+spc*(0.35+la),keep);
        }
        if(uSun>0.0){
          vec2 r2=vec2(p.x*0.82+p.y*0.57,(p.y*0.82-p.x*0.57)*1.5);
          float cl=vn(r2*0.0052+vec2(uTime*0.006,uTime*0.002))*0.72+vn(r2*0.013+vec2(-uTime*0.009,9.0))*0.28;
          float lit=smoothstep(0.33,0.67,cl);
          vec2 fd=min(p-uFld.xy,uFld.xy+uFld.zw-p);
          float inF=uFld.z>0.0?smoothstep(-2.0,8.0,min(fd.x,fd.y)):1.0;
          w*=mix(vec3(1.0),mix(uShCol,uSunCol,lit),uSun*mix(0.45,1.0,inF));
        }
        vec3 L=texture2D(uLight,p/uLow).rgb*2.0;
        vec3 col=w*L;
        float fg=clamp((1.0/gl_FragCoord.w-uFog.x)/(uFog.y-uFog.x),0.0,1.0)*uFog.z;
        gl_FragColor=vec4(mix(col,uFogCol,fg),0.0);
      }`,
    // モンスター（画面と同じ大きさの canvas、乗算済みアルファ）。足もとの地面の明るさを掛ける
    spr3: `
      #ifdef GL_FRAGMENT_PRECISION_HIGH
      precision highp float;
      #else
      precision mediump float;
      #endif
      varying vec2 v;
      uniform sampler2D uSpr,uLight;uniform vec2 uLow,uRes;uniform vec3 uC,uU,uFw;uniform float uF;
      const float SPR_OVER=0.3;
      void main(){
        vec4 s=texture2D(uSpr,vec2(v.x,1.0-v.y));
        if(s.a<0.004)discard;
        float X=(gl_FragCoord.x-uRes.x*0.5)/uF,Y=(gl_FragCoord.y-uRes.y*0.5)/uF;
        vec3 d=vec3(X,uU.y*Y+uFw.y,min(-0.03,uU.z*Y+uFw.z));
        vec2 g=uC.xy+d.xy*(-uC.z/d.z);
        vec3 L=texture2D(uLight,clamp(g/uLow,0.0,1.0)).rgb*2.0;
        // [ユーザー指定 2026-11-16] 強い光源でモンスターが白く飛ばないよう、等倍(1.0)を超えた光は 0.3 倍に圧縮する（最大 2.0 → 1.3）。地面には掛けない。
        L=min(L,1.0)+max(L-1.0,0.0)*SPR_OVER;
        gl_FragColor=vec4(s.rgb*L,s.a);
      }`,
    // 技のエフェクト（地面の面に乗せる。網点と減色は compose と同じ）
    fx3: `
      #ifdef GL_FRAGMENT_PRECISION_HIGH
      precision highp float;
      #else
      precision mediump float;
      #endif
      varying vec2 vP;
      uniform sampler2D uFx;uniform vec2 uLow;
      float b2(vec2 a){a=floor(a);return fract(dot(a,vec2(0.5,a.y*0.75)));}
      float bayer(vec2 a){return b2(0.5*a)*0.25+b2(a);}
      void main(){
        vec2 cell=floor(vP);
        vec4 f=texture2D(uFx,(cell+0.5)/uLow);
        float th=bayer(cell)+0.03;
        if(f.a<0.02||f.a*1.12<th)discard;
        vec3 fc=floor(f.rgb*6.0+0.5)/6.0;
        gl_FragColor=vec4(fc,max(fc.r,max(fc.g,fc.b)));
      }`,
    // [ユーザー指定 2026-11-16] 空中のエフェクトの層（画面の座標、1 ドット = uQ 画素）。網点と減色は fx3 と同じ。モンスターの上に重ねる。
    air3: `
      #ifdef GL_FRAGMENT_PRECISION_HIGH
      precision highp float;
      #else
      precision mediump float;
      #endif
      varying vec2 v;
      uniform sampler2D uAir;uniform vec2 uRes,uAirSize;uniform float uQ;
      float b2(vec2 a){a=floor(a);return fract(dot(a,vec2(0.5,a.y*0.75)));}
      float bayer(vec2 a){return b2(0.5*a)*0.25+b2(a);}
      void main(){
        vec2 cell=floor(vec2(v.x,1.0-v.y)*uRes/uQ);
        vec4 f=texture2D(uAir,(cell+0.5)/uAirSize);
        float th=bayer(cell)+0.03;
        if(f.a<0.02||f.a*1.12<th)discard;
        vec3 fc=floor(f.rgb*6.0+0.5)/6.0;
        gl_FragColor=vec4(fc,max(fc.r,max(fc.g,fc.b)));
      }`,
    bright: COMMON + `
      uniform sampler2D uTex;uniform float uTh,uEm;
      void main(){vec4 c=texture2D(uTex,v);vec3 b=max(c.rgb-uTh,0.0)/(1.0-uTh);gl_FragColor=vec4(b*0.6+c.rgb*c.a*uEm,1.0);}`,
    blur: COMMON + `
      uniform sampler2D uTex;uniform vec2 uDir;
      void main(){
        vec3 c=texture2D(uTex,v).rgb*0.227027;
        c+=texture2D(uTex,v+uDir*1.3846).rgb*0.3162162; c+=texture2D(uTex,v-uDir*1.3846).rgb*0.3162162;
        c+=texture2D(uTex,v+uDir*3.2308).rgb*0.0702703; c+=texture2D(uTex,v-uDir*3.2308).rgb*0.0702703;
        gl_FragColor=vec4(c,1.0);}`,
    bokeh: COMMON + `
      uniform sampler2D uTex;uniform vec2 uDir;
      void main(){
        vec3 acc=vec3(0.0);float ws=0.0;
        vec2 p=vec2(-0.7373688,0.6754903);
        vec2 d=vec2(1.0,0.0);
        for(int i=0;i<12;i++){
          float r=sqrt((float(i)+0.5)/12.0);
          vec3 s=texture2D(uTex,v+d*r*uDir).rgb;
          float l=dot(s,vec3(0.299,0.587,0.114));
          float w=1.0+l*l*l*2.2;
          acc+=s*w;ws+=w;
          d=vec2(d.x*p.x-d.y*p.y,d.x*p.y+d.y*p.x);
        }
        gl_FragColor=vec4(acc/ws,1.0);}`,
    final: `
      #ifdef GL_FRAGMENT_PRECISION_HIGH
      precision highp float;
      #else
      precision mediump float;
      #endif
      varying vec2 v;
      uniform sampler2D uComp,uBloom,uDof,uSprM;uniform float uTime,uBloomK,uTilt,uVig,uSat,uCon,uRays,uGrain,uMote,uSplit,uSprMask;uniform vec3 uRayCol;
      uniform vec3 uTint;uniform vec4 uFog,uFlash,uImp;uniform vec2 uRes;
      float h(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
      float n(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.0-2.0*f);return mix(mix(h(i),h(i+vec2(1,0)),f.x),mix(h(i+vec2(0,1)),h(i+vec2(1,1)),f.x),f.y);}
      float fbm(vec2 p){float s=0.0,a=0.5;for(int i=0;i<4;i++){s+=n(p)*a;p*=2.03;a*=0.5;}return s;}
      void main(){
        vec2 uv=v;float sy=1.0-v.y;
        vec3 c=texture2D(uComp,uv).rgb;
        if(uImp.y>0.0){vec2 dv=(uImp.zw-uv)*uImp.y*0.09;vec3 acc=c;for(int i=1;i<8;i++){acc+=texture2D(uComp,uv+dv*float(i)).rgb;}c=mix(c,acc/8.0,min(1.0,uImp.y*1.4));}
        vec3 d=texture2D(uDof,uv).rgb;
        float sm=uSprMask>0.0?texture2D(uSprM,vec2(v.x,1.0-v.y)).a*uSprMask:0.0; // モンスターがいる画素（0〜1）
        float t=min(1.0,max(smoothstep(0.24,0.05,sy),smoothstep(0.73,0.9,sy))*min(1.15,uTilt)); // 上部遠景と下部手前だけを自然にぼかす（中央〜建物の屋根までは鮮明）
        c=mix(c,d,t*(1.0-0.7*sm)); // モンスターは画面の上端・下端でもぼかしすぎない
        // [ユーザー指定 2026-11-16] モンスターの上では、にじみ（ブルーム）・光の筋・ちりを弱めて体の模様が白く飛ばないようにする（3D のときだけ。uSprMask = 0 で無効）
        c+=texture2D(uBloom,uv).rgb*uBloomK*(1.0-0.6*sm);
        if(uFog.a>0.0){float f=fbm(uv*vec2(3.0,2.2)+vec2(uTime*0.025,uTime*0.01));c=mix(c,uFog.rgb,smoothstep(0.35,0.9,f)*uFog.a);}
        // [ユーザー指定 2026-11-16] 光の筋: 左上から右下へ差しこむ、幅の広いやわらかい光の束。筋ごとに太さと明るさが違い、ゆっくりゆらぐ。
        // 光源（左上）に近いほど強く、あわせて薄いもやを足す（空気感）。
        if(uRays>0.0){
          float a=uv.x*0.86-sy*0.52;
          float s1=n(vec2(a*4.2+11.0,uTime*0.035)),s2=n(vec2(a*9.5+3.0,uTime*0.06+5.0)),s3=n(vec2(a*21.0,uTime*0.09+9.0));
          float sh=smoothstep(0.34,0.66,s1*0.55+s2*0.3+s3*0.15);
          float fall=clamp(1.15-(uv.x*0.55+sy*0.62),0.0,1.0);fall=fall*fall*(3.0-2.0*fall);
          c+=uRayCol*(sh*fall*0.5+fall*0.04)*uRays*(1.0-0.75*sm);
        }
        // 光の中をただよう細かいちり
        if(uMote>0.0){
          float mt=0.0;vec2 asp=vec2(uRes.x/uRes.y,1.0);
          for(int i=0;i<2;i++){
            float L=i==0?9.0:15.0;vec2 g=uv*asp*L+vec2(uTime*(i==0?0.05:-0.035),uTime*(i==0?0.03:0.045))+float(i)*7.3;
            vec2 id=floor(g),f=fract(g)-0.5;float r=h(id);
            if(r>0.8){vec2 o=(vec2(h(id+1.7),h(id+5.3))-0.5)*0.6;float d=length(f-o);mt+=smoothstep(i==0?0.07:0.05,0.0,d)*max(0.0,0.45+0.55*sin(uTime*1.7+r*40.0));}
          }
          c+=uRayCol*mt*uMote*(0.35+0.65*clamp(1.1-(uv.x*0.6+sy*0.7),0.0,1.0))*(1.0-0.75*sm);
        }
        c*=uTint;
        float l=dot(c,vec3(0.299,0.587,0.114));
        // 暗い所は少し青く、明るい所は少し黄色く（夕方の光の色分け）
        if(uSplit>0.0)c*=mix(vec3(1.0),mix(vec3(0.93,0.98,1.08),vec3(1.07,1.03,0.92),smoothstep(0.18,0.72,l)),uSplit);
        c=mix(vec3(l),c,uSat);
        c=(c-0.5)*uCon+0.5;
        vec2 q=uv-0.5;c*=1.0-uVig*dot(q,q)*1.6;
        c=mix(c,uFlash.rgb,uFlash.a);
        if(uImp.x>0.5){float g=dot(c,vec3(0.299,0.587,0.114));c=vec3(1.0)-vec3(g*0.9,g*0.95,g*0.75);}
        c+=(h(gl_FragCoord.xy+fract(uTime)*100.0)-0.5)*uGrain;
        gl_FragColor=vec4(c,1.0);
      }`,
  };

  function shader(type, src) {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  // 3D の地面用: アリーナの絵（0..uLow）を z = 0 の面として置く
  const VS3 = "attribute vec2 p;uniform mat4 uVP;uniform vec2 uLow;varying vec2 vP;void main(){vec2 g=(p*0.5+0.5)*uLow;vP=g;gl_Position=uVP*vec4(g,0.,1.);}";
  function program(fs, vs) {
    const p = gl.createProgram();
    gl.attachShader(p, shader(gl.VERTEX_SHADER, vs || VS)); gl.attachShader(p, shader(gl.FRAGMENT_SHADER, fs));
    gl.bindAttribLocation(p, 0, "p"); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i); u[a.name] = gl.getUniformLocation(p, a.name); }
    return { p, u };
  }
  function texture(filter) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  function target(w, h) {
    const t = texture(gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    const f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    return { t, f, w, h };
  }

  function init(canvas) {
    try {
      gl = canvas.getContext("webgl", { alpha: false, antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: true });
      if (!gl) return false;
      for (const k in FS) P[k] = program(FS[k], k === "ground3" || k === "fx3" ? VS3 : null);
      if (typeof Scene3D !== "undefined") { try { Scene3D.initGL(gl); } catch (e) { console.warn("Scene3D disabled:", e); Scene3D.disable(); } }
      quad = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, quad);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      T.world = texture(gl.LINEAR); /* 中心を読めば元のドットそのまま。HD 加工は中間も読む */ T.fx = texture(gl.NEAREST); T.air = texture(gl.NEAREST); T.light = texture(gl.LINEAR); T.spr = texture(gl.LINEAR);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
      ok = true;
    } catch (e) { console.warn("WebGL post disabled:", e); ok = false; }
    return ok;
  }

  function resize(w, h) {
    if (!ok || (w === W && h === H)) return;
    W = w; H = h; QW = Math.max(1, Math.round(w / 4)); QH = Math.max(1, Math.round(h / 4));
    for (const k in F) { gl.deleteTexture(F[k].t); gl.deleteFramebuffer(F[k].f); if (F[k].d) gl.deleteRenderbuffer(F[k].d); }
    F.comp = target(W, H);
    // 3D の立体どうしの前後関係のための深度バッファ（comp だけ）
    F.comp.d = gl.createRenderbuffer(); gl.bindRenderbuffer(gl.RENDERBUFFER, F.comp.d);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_STENCIL, W, H);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, F.comp.d); F.a = target(QW, QH); F.b = target(QW, QH); F.c = target(QW, QH); F.d = target(QW, QH);
  }

  function upload(t, canvas) { gl.bindTexture(gl.TEXTURE_2D, t); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas); }
  function pass(prog, out, setup) {
    gl.useProgram(prog.p);
    if (out) { gl.bindFramebuffer(gl.FRAMEBUFFER, out.f); gl.viewport(0, 0, out.w, out.h); }
    else { gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, W, H); }
    setup(prog.u);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
  function bind(unit, tex, loc) { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(loc, unit); }

  // 3D の合成: 空の色 → 地面（world）→ 奥の立体 → モンスター → 技 → 手前の立体。結果は comp（rgb + 発光 a）で、以降のブルーム等は共通。
  function compose3d(o, g) {
    const V = Scene3D.view, sc = Scene3D.use(o.three.A), c = sc.cfg, fogCol = o.three.fogCol || c.fog.color;
    const quadAttr = () => { gl.bindBuffer(gl.ARRAY_BUFFER, quad); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0); };
    gl.bindFramebuffer(gl.FRAMEBUFFER, F.comp.f); gl.viewport(0, 0, W, H);
    const cc = o.three.clear || c.clear;
    gl.clearColor(cc[0], cc[1], cc[2], 0); gl.clearDepth(1); gl.depthMask(true);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL);
    pass(P.ground3, F.comp, (u) => {
      bind(0, T.world, u.uWorld); bind(2, T.light, u.uLight);
      gl.uniformMatrix4fv(u.uVP, false, V.vp); gl.uniform2f(u.uLow, o.lowW, o.lowH); gl.uniform1f(u.uF, V.F); gl.uniform1f(u.uTime, o.time);
      const hd = o.hd || [0, 0, 0, 0, 1, 1, 30];
      gl.uniform4f(u.uHd, hd[0], hd[1], hd[2], hd[3]); gl.uniform3f(u.uMat, hd[4], hd[5], hd[6]);
      gl.uniform4fv(u.uFld, o.fld || [0, 0, 0, 0]); gl.uniform1f(u.uSun, g.sun || 0); gl.uniform3fv(u.uSunCol, g.sunCol || [1.2, 1.14, 0.9]); gl.uniform3fv(u.uShCol, g.shCol || [0.66, 0.76, 0.9]);
      const cl = c.clip; gl.uniform4f(u.uClip, cl ? cl.x : 0, cl ? cl.y : 0, cl ? cl.w : 0, cl ? cl.h : 0); gl.uniform1f(u.uClipR, cl ? cl.r || 0 : 0);
      gl.uniform3f(u.uFog, V.d + c.fog.near, V.d + c.fog.far, c.fog.max); gl.uniform3fv(u.uFogCol, fogCol);
    });
    const so = { lightTex: T.light, lowW: o.lowW, lowH: o.lowH, time: o.time, hype: o.three.hype || 0, on: o.three.on, fogCol };
    Scene3D.draw("back", so);
    quadAttr();
    // 地面のエフェクト（予告範囲・輪・影）→ モンスター → 空中のエフェクト（立っている層）の順に重ねる
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
    pass(P.fx3, F.comp, (u) => { bind(1, T.fx, u.uFx); gl.uniformMatrix4fv(u.uVP, false, V.vp); gl.uniform2f(u.uLow, o.lowW, o.lowH); });
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ZERO, gl.ONE_MINUS_SRC_ALPHA);
    pass(P.spr3, F.comp, (u) => {
      bind(0, T.spr, u.uSpr); bind(2, T.light, u.uLight);
      gl.uniform2f(u.uLow, o.lowW, o.lowH); gl.uniform2f(u.uRes, W, H); gl.uniform1f(u.uF, V.F);
      gl.uniform3fv(u.uC, V.C); gl.uniform3fv(u.uU, V.u); gl.uniform3fv(u.uFw, V.f);
    });
    gl.disable(gl.BLEND);
    if (o.air) pass(P.air3, F.comp, (u) => { bind(1, T.air, u.uAir); gl.uniform2f(u.uRes, W, H); gl.uniform2f(u.uAirSize, o.air.width, o.air.height); gl.uniform1f(u.uQ, o.airQ); });
    gl.enable(gl.DEPTH_TEST);
    Scene3D.draw("front", so);
    quadAttr();
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
  }

  // o: { world, fx, light, lowW, lowH, S, fracX, fracY, zoom, time, grade:{tint,sat,con,vig,bloom,tilt,rays,fog,grain}, flash:[r,g,b,a] }
  function render(o) {
    upload(T.world, o.world); upload(T.fx, o.fx); upload(T.light, o.light); if (o.air) upload(T.air, o.air);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true); upload(T.spr, o.spr); gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    const g = o.grade;
    if (o.three) compose3d(o, g); else pass(P.compose, F.comp, (u) => {
      bind(0, T.world, u.uWorld); bind(1, T.fx, u.uFx); bind(2, T.light, u.uLight); bind(3, T.spr, u.uSpr);
      gl.uniform2f(u.uLow, o.lowW, o.lowH); gl.uniform2f(u.uSprLow, o.sprW, o.sprH); gl.uniform2f(u.uRes, W, H); gl.uniform2f(u.uFrac, o.fracX, o.fracY);
      gl.uniform1f(u.uS, o.S); gl.uniform1f(u.uZoom, o.zoom); gl.uniform1f(u.uPad, o.pad || 1); gl.uniform2f(u.uSprOff, o.sprX || 0, o.sprY || 0); gl.uniform1f(u.uSprK, o.sprK || 2); if (u.uTime) gl.uniform1f(u.uTime, o.time);
      const hd = o.hd || [0, 0, 0, 0, 1, 1, 30];
      gl.uniform4f(u.uHd, hd[0], hd[1], hd[2], hd[3]); gl.uniform3f(u.uMat, hd[4], hd[5], hd[6]); gl.uniform2f(u.uOrg, o.orgX || 0, o.orgY || 0);
      gl.uniform4fv(u.uFld, o.fld || [0, 0, 0, 0]); gl.uniform1f(u.uSun, g.sun || 0); gl.uniform3fv(u.uSunCol, g.sunCol || [1.2, 1.14, 0.9]); gl.uniform3fv(u.uShCol, g.shCol || [0.66, 0.76, 0.9]);
    });
    pass(P.bright, F.a, (u) => { bind(0, F.comp.t, u.uTex); gl.uniform1f(u.uTh, g.th || 0.78); gl.uniform1f(u.uEm, g.em || 0.9); });
    const blur = (src, dst, dx, dy) => pass(P.blur, dst, (u) => { bind(0, src.t, u.uTex); gl.uniform2f(u.uDir, dx / QW, dy / QH); });
    blur(F.a, F.b, 1.5, 0); blur(F.b, F.a, 0, 1.5); blur(F.a, F.b, 3.5, 0); blur(F.b, F.a, 0, 3.5);
    pass(P.blur, F.c, (u) => { bind(0, F.comp.t, u.uTex); gl.uniform2f(u.uDir, 1.5 / QW, 0); });
    blur(F.c, F.d, 0, 1.5);
    if (g.tilt > 0) {
      const r = g.tilt > 1 ? 3.2 : 2.2;
      pass(P.bokeh, F.c, (u) => { bind(0, F.d.t, u.uTex); gl.uniform2f(u.uDir, r / QW, r / QH); });
      blur(F.c, F.d, 1.1, 0); blur(F.d, F.c, 0, 1.1);
      pass(P.bokeh, F.d, (u) => { bind(0, F.c.t, u.uTex); gl.uniform2f(u.uDir, (r * 0.55) / QW, (r * 0.55) / QH); });
    }
    pass(P.final, null, (u) => {
      bind(0, F.comp.t, u.uComp); bind(1, F.a.t, u.uBloom); bind(2, F.d.t, u.uDof); bind(3, T.spr, u.uSprM); gl.uniform1f(u.uSprMask, o.three ? 1 : 0);
      gl.uniform1f(u.uTime, o.time); gl.uniform1f(u.uBloomK, g.bloom); gl.uniform1f(u.uTilt, g.tilt);
      gl.uniform1f(u.uVig, g.vig); gl.uniform1f(u.uSat, g.sat); gl.uniform1f(u.uCon, g.con); gl.uniform1f(u.uRays, g.rays || 0);
      gl.uniform1f(u.uGrain, g.grain === undefined ? 0.03 : g.grain);
      gl.uniform1f(u.uMote, g.mote || 0); gl.uniform1f(u.uSplit, g.split || 0); gl.uniform3fv(u.uRayCol, g.rayCol || [1.0, 0.95, 0.8]);
      gl.uniform3fv(u.uTint, g.tint); gl.uniform4fv(u.uFog, g.fog || [0, 0, 0, 0]); gl.uniform4fv(u.uFlash, o.flash); gl.uniform4fv(u.uImp, o.imp || [0, 0, 0.5, 0.5]);
      gl.uniform2f(u.uRes, W, H);
    });
  }

  return { init, resize, render, ok: () => ok };
})();

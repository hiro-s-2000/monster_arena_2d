import json,os,sys,struct,shutil,hashlib
from PIL import Image, ImageOps, ImageSequence
W='/Users/honmahiroshi/Desktop/ai_workspace'; P=W+'/arena_revenge_claude'; R=P+'/monsterSprite'
BK=W+'/_backup_monsterSprite_gifs_before_pokestats_20261115'
DRY='--apply' not in sys.argv
md5=lambda p: hashlib.md5(open(p,'rb').read()).hexdigest()

# ---- lossless GIF horizontal flip (keeps palettes, delays, disposal; only pixel rows and x offsets change) ----
def subblocks(d,i):
    out=bytearray()
    while d[i]: out+=d[i+1:i+1+d[i]]; i+=d[i]+1
    return bytes(out),i+1
def lzw_dec(data,mcs,n):
    clear=1<<mcs; end=clear+1; size=mcs+1; nxt=end+1
    pre=[0]*4096; suf=[0]*4096
    for k in range(clear): suf[k]=k
    out=bytearray(); bits=0; nb=0; prev=-1; first=0; stack=[]
    for byte in data:
        bits|=byte<<nb; nb+=8
        while nb>=size:
            code=bits&((1<<size)-1); bits>>=size; nb-=size
            if code==clear: size=mcs+1; nxt=end+1; prev=-1; continue
            if code==end: return bytes(out[:n])
            if prev==-1: out.append(suf[code]); prev=code; first=code; continue
            cur=code
            if code>=nxt: stack.append(first); cur=prev
            while cur>=clear: stack.append(suf[cur]); cur=pre[cur]
            first=suf[cur]; stack.append(first)
            if nxt<4096:
                pre[nxt]=prev; suf[nxt]=first; nxt+=1
                if nxt==(1<<size) and size<12: size+=1
            prev=code
            out.extend(reversed(stack)); stack.clear()
            if len(out)>=n: return bytes(out[:n])
    return bytes(out[:n])
def lzw_enc(px,mcs):
    clear=1<<mcs; end=clear+1; size=mcs+1; nxt=end+1; table={}
    out=bytearray(); bits=0; nb=0
    def emit(c):
        nonlocal bits,nb
        bits|=c<<nb; nb+=size
        while nb>=8: out.append(bits&255); bits>>=8; nb-=8
    emit(clear); cur=px[0]
    for p in px[1:]:
        key=(cur<<8)|p
        if key in table: cur=table[key]
        else:
            emit(cur)
            if nxt<4096:
                table[key]=nxt; nxt+=1
                if nxt>(1<<size) and size<12: size+=1
            else:
                emit(clear); table={}; size=mcs+1; nxt=end+1
            cur=p
    emit(cur); emit(end)
    if nb: out.append(bits&255)
    res=bytearray()
    for i in range(0,len(out),255): ch=out[i:i+255]; res.append(len(ch)); res+=ch
    res.append(0); return bytes(res)
def flip_gif(src,dst):
    d=open(src,'rb').read(); cw=struct.unpack('<H',d[6:8])[0]; i=13
    if d[10]&0x80: i+=3*(2<<(d[10]&7))
    out=bytearray(d[:i])
    while i<len(d):
        b=d[i]
        if b==0x3B: out.append(b); break
        if b==0x21:
            j=i+2
            while d[j]: j+=d[j]+1
            j+=1; out+=d[i:j]; i=j
        elif b==0x2C:
            l,t,w,h=struct.unpack('<HHHH',d[i+1:i+9]); fl=d[i+9]; j=i+10
            lct=b''
            if fl&0x80: n=3*(2<<(fl&7)); lct=d[j:j+n]; j+=n
            mcs=d[j]; data,j2=subblocks(d,j+1)
            px=lzw_dec(data,mcs,w*h); assert len(px)==w*h,(src,len(px),w*h)
            fp=bytearray()
            for y in range(h): fp+=px[y*w:(y+1)*w][::-1]
            out.append(0x2C); out+=struct.pack('<HHHH',cw-l-w,t,w,h); out.append(fl); out+=lct; out.append(mcs); out+=lzw_enc(bytes(fp),mcs)
            i=j2
        else: raise ValueError((src,i,b))
    os.makedirs(os.path.dirname(dst),exist_ok=True); open(dst,'wb').write(out)

def frames(p):
    im=Image.open(p); fr=[]
    for f in ImageSequence.Iterator(im): fr.append((f.convert('RGBA').tobytes(), f.info.get('duration',0)))
    return im.size,fr
def is_flip(a,b):
    try:
        sa,fa=frames(a); im=Image.open(b)
        if im.size!=sa or getattr(im,'n_frames',1)!=len(fa): return False
        for k,f in enumerate(ImageSequence.Iterator(im)):
            if ImageOps.mirror(f.convert('RGBA')).tobytes()!=fa[k][0] or f.info.get('duration',0)!=fa[k][1]: return False
        return True
    except Exception as e: return False
def bounds(p):
    im=Image.open(p); im.seek(0); a=im.convert('RGBA'); bb=a.getchannel('A').point(lambda v:255 if v else 0).getbbox()
    return [im.size[0],im.size[1]]+([bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1]] if bb else [0,0,im.size[0],im.size[1]])
def backup(rel):
    src=R+'/'+rel
    if os.path.exists(src) and not DRY:
        dst=BK+'/'+rel
        if not os.path.exists(dst): os.makedirs(os.path.dirname(dst),exist_ok=True); shutil.copy2(src,dst)

def main():
  forms=json.load(open('forms.json')); res=[]; newb={}
  for f in forms:
    r=dict(slug=f['slug'],dir=f['dir'],id=f['id']); res.append(r)
    if f['slug']=='latias-mega': r['skip']='locked spec 3'; continue
    for side in ('front','back'):
        src=f"dl/{side}/{f['id']}.gif"
        if not os.path.exists(src): r[side]='no source: untouched'; continue
        rel=f"{f['dir']}/{side}.gif"; relr=f"{f['dir']}/{side}-reverse.gif"; loc=R+'/'+rel; locr=R+'/'+relr
        act=[]
        if not os.path.exists(loc): act.append('created')
        elif md5(loc)!=md5(src): act.append('replaced')
        if act:
            backup(rel)
            if not DRY: shutil.copyfile(src,loc); newb[rel]=bounds(loc)
        base=src if DRY else loc
        if act or not os.path.exists(locr) or not is_flip(base,locr):
            act.append('reverse created' if not os.path.exists(locr) else 'reverse rewritten')
            backup(relr)
            if not DRY:
                flip_gif(loc,locr); assert is_flip(loc,locr),locr; newb[relr]=bounds(locr)
        r[side]=', '.join(act) if act else 'already identical'
  json.dump(res,open('result.json' if not DRY else 'result_dry.json','w'),ensure_ascii=False,indent=0)
  import collections
  c=collections.Counter((s,r.get(s,r.get('skip'))) for r in res for s in ('front','back'))
  for k in sorted(c,key=str): print(k,c[k])
  if not DRY:
    bp=P+'/data/sprite_bounds.js'; raw=open(bp,encoding='utf-8').read(); i=raw.index('{'); j=raw.rindex('}')+1
    B=json.loads(raw[i:j]); ch=0
    for k,v in newb.items():
        if B['g'].get(k)!=v: B['g'][k]=v; ch+=1
    open(bp,'w',encoding='utf-8').write(raw[:i]+json.dumps(B,separators=(',',':'),ensure_ascii=False)+raw[j:])
    print('bounds entries updated',ch,'of',len(newb))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Identity-preserving Venusaur/Blastoise Gmax actions from original front/back pixels.
Raichu X workflow: locked palette, anatomical layers, 4x nearest posing, gait QC.
Two source angles + mirrors, deliberately no invented/flattened side views.
"""
import argparse, hashlib, importlib.util, json, math, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parents[1]
FORGE=ROOT.parent/'agent-sprite-forge'; PY=FORGE/'.venv/bin/python'
def module(n):
 s=importlib.util.spec_from_file_location(n,ROOT/'tools'/f'{n}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ANATOMY=module('forge_anatomical_actions')
CW,CH,FY=224,160,128
N=16; ACTS=['idle','walk','attack','hurt'];DIRS=['S','SE','E','NE','N','NW','W','SW'];AD=[.085,.075,.065,.065]
SRC=[('F',False),('F',True),('F',True),('B',False),('B',False),('B',True),('F',False),('F',False)]
VERSION='original_rig_v4'
def polygon(points,shape):
 im=Image.new('L',(shape[1],shape[0]));ImageDraw.Draw(im).polygon(points,fill=255);return np.array(im)>0

def rig(a,name,view):
 y,x=np.indices(a.shape[:2]);op=a[...,3]>0;r,g,b=[a[...,i].astype(int) for i in range(3)]
 masks={};piv={}
 def part(n,m,p):masks[n]=m&op;piv[n]=p
 if name=='venusaur':
  part('body',op,(48,78))
  part('leg_far',(y>=76)&(x<36),(23,77))
  part('leg_near',(y>=76)&(x>=48)&(x<84),(65,77))
  if view=='B':part('leg_rear',(y>=75)&(x>=82),(87,76))
  if view=='F':part('head',polygon([(8,53),(48,53),(60,63),(52,78),(25,80),(6,70)],op.shape),(42,62))
  part('leaf_far',(x<19)&(y>=54)&(y<77),(14,57))
  part('leaf_near',(x>=52)&(y>=53)&(y<81)&(g>r)&(g>b*1.3),(65,56))
  flower=(y<60)&(r>g*1.15)&(r>b*1.12)
  flower=ndi.binary_dilation(flower,iterations=1)&op&(y<61)
  part('flower',flower,(47,49))
  part('crown',(y<32)&(r>g*.95)&(g>b*1.4),(49,27))
 else:
  part('shell',op,(54,57))
  part('body',polygon([(12,38),(61,38),(75,66),(68,81),(22,81),(10,56)],op.shape)|(op&(r>g+35)&(g>b+20)),(48,77))
  part('leg_far',(y>=74)&(x>=10)&(x<39),(25,77))
  part('tail',(y>=72)&((x>=80) if view=='F' else (x<40)),(82,78) if view=='F' else (35,80))
  part('leg_near',(y>=63)&(x>=49)&(x<82),(65,69))
  if view=='F':
   part('head',polygon([(0,15),(30,15),(35,30),(32,46),(9,48),(0,40)],op.shape),(25,39))
   part('arm_far',(x<12)&(y>=45)&(y<64),(8,47))
   part('arm_near',polygon([(36,32),(56,31),(63,43),(58,58),(43,58),(37,46)],op.shape),(46,35))
  else:
   part('head',(x>=67)&(y>=15)&(y<36),(75,29))
   part('arm_near',(x>=80)&(y>=34)&(y<58),(83,37))
  # Actual armour pixels always remain with the rigid shell, never attached to a limb.
  armour=(r>g*.95)&(g>b*1.08)&((r-g)<35)&(y<77)
  if view=='F':
   belly=polygon([(12,36),(36,36),(45,47),(49,54),(62,63),(60,77),(39,82),(17,75),(9,59),(9,47)],op.shape)
   armour &= ~belly
  rim=(np.minimum(np.minimum(r,g),b)>175)&(y<76)
  masks['shell']|=armour|rim
  for n in masks:
   if n!='shell':masks[n]&=~(armour|rim)
 # Last named layer owns each original pixel exactly once.
 names=list(masks);lab=np.zeros(op.shape,int)
 for i,n in enumerate(names):lab[masks[n]]=i+1
 masks={n:lab==i+1 for i,n in enumerate(names)}
 layers=ANATOMY.build_ordered_layers(a,list(masks.items()),cushion_px=5)
 if name=='blastoise':
  # The white shell rim must not be extended underneath the moving orange belly.
  seam=(layers['shell'][...,3]>0)&~masks['shell']&masks['body']
  orange=op&(r>g+35)&(g>b+20)&(r>110)
  if seam.any() and orange.any():
   _,(iy,ix)=ndi.distance_transform_edt(~orange,return_indices=True)
   layers['shell'][seam]=a[iy[seam],ix[seam]]
 return layers,piv,masks

def curve(f,values):return values[f]
def motion(name,view,act,f):
 p={};phase=2*math.pi*f/N
 if act=='idle':
  breath=math.sin(phase)
  p['body']=(0,0,-.8*breath)
  p['head']=(0,0,-.8*breath)
  if name=='venusaur':
   p['flower']=(.7*breath,0,-1.1*breath);p['crown']=(1.6*math.sin(phase),0,-1.1*breath)
   p['leaf_near']=(1.5*breath,0,-.6*breath);p['leaf_far']=(-1.5*breath,0,-.6*breath)
  else:
   p['shell']=(0,0,-.7*breath);p['arm_near']=(1.5*breath,0,-.5*breath);p['tail']=(2*breath,0,0)
 elif act=='walk':
  # One full 16-frame stride: two opposing steps, grounded stance then lifted return.
  for n,offset in [('leg_near',0),('leg_far',.5),('leg_rear',.5)]:
   dx,dy,deg=ANATOMY.stride_arc(f/N+offset,3.0 if name=='venusaur' else 3.5,2.5 if name=='venusaur' else 3.5)
   p[n]=(deg*.20,-dx,dy)
  bob=.65*math.sin(phase)
  p['body']=(0,0,bob);p['head']=(0,.4*math.sin(phase),bob)
  if name=='venusaur':
   p['flower']=(1.1*math.sin(phase),0,bob*.6)
   p['crown']=(2*math.sin(phase-.4),0,bob*.6)
   p['leaf_near']=(3*math.sin(phase),0,bob);p['leaf_far']=(-3*math.sin(phase),0,bob)
  else:
   p['shell']=(.8*math.sin(phase),0,bob*.6)
   p['arm_near']=(6*math.sin(phase),0,bob);p['arm_far']=(-6*math.sin(phase),0,bob)
   p['tail']=(4*math.sin(phase-.5),0,0)
 elif act=='attack':
  if name=='venusaur':
   load=curve(f,[0,.3,.65,1,1,.6,0,0,0,0,0,0,0,0,0,0]);strike=curve(f,[0,0,0,0,0,.4,1,1,1,.8,.5,.25,.1,0,0,0])
   p['body']=(0,-1.5*load+2*strike,1.5*load)
   p['head']=(0,-load+3*strike,load)
   p['flower']=(-2*load+2.5*strike,-load+2*strike,2*load)
   p['crown']=(-2*load+3*strike,-load+2*strike,2*load)
   p['leg_near']=(0,2*strike,-1.5*load);p['leg_far']=(0,strike,-load)
   p['leaf_near']=(4*strike,2*strike,load);p['leaf_far']=(-4*strike,strike,load)
  else:
   load=curve(f,[0,.3,.7,1,1,1,.4,.25,.25,.15,.1,0,0,0,0,0]);recoil=curve(f,[0,0,0,0,0,0,3,2,3,2,1.5,1,.5,0,0,0])
   p['body']=(0,-recoil*.6,load);p['head']=(0,-recoil*.5,load)
   p['shell']=(0,-recoil,1.5*load);p['arm_near']=(-5*load,-recoil*.4,load);p['arm_far']=(4*load,-recoil*.4,load)
   p['tail']=(3*recoil,0,0)
 elif act=='hurt':
  e=curve(f,[0,1,.85,.7,.6,.45,.3,.2,.1,0,0,0,0,0,0,0]);j=(1 if f%2 else -1)*e*.4
  for n in ['body','head','shell','flower','crown','arm_near','arm_far','leaf_near','leaf_far']:
   p[n]=(0,-3*e+j,-e)
  p['tail']=(4*e,0,0)
 return p

def transform(a,pivot,pose):
 angle,dx,dy=pose
 hi=Image.fromarray(a).resize((CW*4,CH*4),Image.Resampling.NEAREST)
 if angle:hi=hi.rotate(-angle,Image.Resampling.NEAREST,center=(pivot[0]*4+2,pivot[1]*4+2))
 out=Image.new('RGBA',hi.size);out.paste(hi,(round(dx*4),round(dy*4)))
 return np.array(out)[1::4,1::4].copy()

def fx(a,name,view,f,points,colors):
 # Palette-only effects. Body/head pixels are never blurred or recoloured.
 im=Image.fromarray(a);d=ImageDraw.Draw(im)
 def color(target):return min(colors,key=lambda c:sum((c[i]-target[i])**2 for i in range(3)))
 bright=color((225,250,255));water=color((65,175,225));deep=color((20,90,155))
 pink=color((255,125,140));petalshade=color((210,65,65));petalshine=color((255,185,185))
 direction=-1 if view=='F' else 1
 def line(pts,col,w=1):d.line([(round(x),round(y)) for x,y in pts],fill=(*col,255),width=w)
 if name=='blastoise':
  for i,(x,y) in enumerate(points):
   age=f-6-(i%2)
   if 2<=f<=5:
    radius=9-(f-2)*2
    for j in range(5):
     t=j*math.tau/5+f*.5
     line([(x+math.cos(t)*radius,y+math.sin(t)*radius),(x+math.cos(t)*(radius+2),y+math.sin(t)*(radius+2))],water)
   if 0<=age<=6:
    length=[11,23,36,46,49,38,21][age]
    path=[]
    for k in range(length):
     u=k/max(1,length-1)
     # Continuous pressure jet with a slightly falling tip and a foamy breakup.
     px=x+direction*k*.9;py=y-k*.42+u*u*3
     path.append((px,py))
    line(path,deep,7 if i==0 else 5)
    line(path,water,5 if i==0 else 3)
    line([(px,py-1) for px,py in path],bright,2 if i==0 else 1)
    ex,ey=path[-1]
    for j in range(5):
     t=j*math.tau/5+f*.2;r=3+age*.5
     px=ex+math.cos(t)*r;py=ey+math.sin(t)*r
     line([(px,py),(px+direction*2,py+1)],bright if j%2 else water,2)
    # Alternating pressure bands travel along the stream, rather than random flicker.
    for k in range((f*3)%9,len(path),9):
     px,py=path[k];line([(px-1,py-2),(px+1,py+2)],bright)
 else:
  x,y=points[0]
  if 1<=f<=5:
   for j in range(6):
    t=j*math.tau/6+f*.2;r=17-(f-1)*3
    line([(x+math.cos(t)*r,y+math.sin(t)*r),(x+math.cos(t)*(r-2),y+math.sin(t)*(r-2))],petalshine)
  if 6<=f<=13:
   # Eight readable, outlined petals spiral out of the canopy edge and travel forward.
   for j in range(8):
    age=f-6-(j%3)*.35
    if age<0:continue
    t=j*2.4+age*.45
    px=x+direction*(20+age*5)+(j%2*2-1)*math.cos(t)*10
    py=y+20+math.sin(t)*14+age*1.4
    ang=t*.5;cs=math.cos(ang);sn=math.sin(ang)
    def pt(u,v):return (round(px+u*cs-v*sn),round(py+u*sn+v*cs))
    d.polygon([pt(-4,0),pt(0,-2),pt(4,0),pt(0,2)],fill=(*petalshade,255))
    d.polygon([pt(-2,0),pt(1,-1),pt(3,0),pt(0,1)],fill=(*pink,255))
    line([pt(0,-1),pt(2,0)],petalshine)
 return np.array(im)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--only',choices=['venusaur','blastoise']);ap.add_argument('--publish',action='store_true');args=ap.parse_args()
 for dex,name in [(3,'venusaur'),(9,'blastoise')]:
  if args.only and args.only!=name:continue
  key=f'{dex:04d}_{name}-gmax';rel=f'monsterSprite/gen1/{dex:04d}_{name}/gigantamax/{name}-gmax';form=ROOT/rel
  out=form/'agent-sprite-forge';work=out/'_original_v4';work.mkdir(parents=True,exist_ok=True)
  raw=work/'sources';raw.mkdir(exist_ok=True)
  sources={};mapping={'F':('front.png',False),'B':('back.png',False)} if name=='venusaur' else {'F':('back.png',True),'B':('front.png',True)}
  for v,(file,flip) in mapping.items():
   a=np.array(Image.open(form/file).convert('RGBA'));a=a[:,::-1].copy() if flip else a;a[a[...,3]==0]=0;sources[v]=a;Image.fromarray(a).save(raw/f'{v}.png')
  cols=sorted({tuple(map(int,c)) for a in sources.values() for c in np.unique(a[a[...,3]>0,:3],axis=0)})
  pal=work/'palette';lock=work/'palette_lock'
  for p in [pal,lock]:
   if p.exists():shutil.rmtree(p)
  tool=FORGE/'skills/generate2dsprite/scripts/palette_tool.py'
  def run(params):subprocess.run([str(PY),'-I',str(tool),*params],check=True,stdout=subprocess.DEVNULL)
  run(['build','--input',str(raw),'--colors',str(len(cols)),'--reserve',*['#%02x%02x%02x'%c for c in cols],'--name',key,'--output-dir',str(pal)])
  run(['lock','--palette',str(pal/'palette.json'),'--source',str(raw),'--output-dir',str(lock)])
  applied=work/'locked_sources'
  if applied.exists():shutil.rmtree(applied)
  run(['apply','--palette',str(lock/'palette.json'),'--input',str(raw),'--output-dir',str(applied)])
  frames={};diff={};qa={'off_palette_colors':0,'intermediate_alpha':0,'edge_touching_frames':[],'static_diff_pixels':diff,'source_route':'existing original front.png/back.png; no AI redraw','direction_limit':'two source angles plus mirrors; no fabricated side projection'}
  for view,a in sources.items():
   locked=np.array(Image.open(applied/f'{view}.png').convert('RGBA'));assert np.array_equal(a[a[...,3]>0],locked[a[...,3]>0]),'palette changed source pixels'
   layers,piv,masks=rig(a,name,view);ys,xs=np.nonzero(a[...,3]);ox=CW//2-48;oy=FY-int(ys.max());ref=np.zeros((CH,CW,4),np.uint8);ref[oy:oy+96,ox:ox+96]=a
   cells={}
   for n,layer in layers.items():
    cell=np.zeros_like(ref);cell[oy:oy+96,ox:ox+96]=layer;cells[n]=cell;piv[n]=(piv[n][0]+ox,piv[n][1]+oy)
   for act in ACTS:
    seq=[];dest=work/'frames'/view/act;dest.mkdir(parents=True,exist_ok=True)
    for f in range(N):
     poses=motion(name,view,act,f);canvas=np.zeros_like(ref)
     for n,cell in cells.items():
      p=poses.get(n,(0,0,0));direction=-1 if view=='F' else 1
      layer=transform(cell,piv[n],(p[0]*(-direction),p[1]*direction,p[2]));m=layer[...,3]>0;canvas[m]=layer[m]
     if act=='attack':
      if name=='venusaur':points=[(ox+49,oy+17)]
      else:
       points=[(ox+78,oy+10),(ox+63,oy+25)] if view=='F' else [(ox+29,oy+22),(ox+14,oy+40)]
       p=poses.get('shell',(0,0,0));points=[(x+p[1]*direction,y+p[2]) for x,y in points]
      canvas=fx(canvas,name,view,f,points,cols)
     if act=='hurt' and f==1:
      # Palette-preserving rim flash leaves the face and species details legible.
      op=canvas[...,3]>0;edge=op&~ndi.binary_erosion(op);canvas[edge,:3]=max(cols,key=sum)
     canvas[canvas[...,3]==0]=0
     if act=='idle' and f==0:diff[view]=int(np.any(canvas!=ref,axis=2).sum())
     op=canvas[...,3]>0
     if op[0].any() or op[-1].any() or op[:,0].any() or op[:,-1].any():qa['edge_touching_frames'].append(f'{view}/{act}/{f}')
     assert set(map(tuple,np.unique(canvas[op,:3],axis=0)))<=set(cols)
     Image.fromarray(canvas).save(dest/f'{f:02d}.png');seq.append(canvas)
    frames[(view,act)]=seq
   # Explicit labelled rig inspection image.
   label=np.zeros_like(a);colors=[(160,80,60),(30,160,80),(60,100,240),(220,120,30),(220,50,160),(50,210,210),(230,210,70),(100,100,100),(200,80,80)]
   for i,(n,m) in enumerate(masks.items()):label[m]=(*colors[i%len(colors)],255)
   Image.fromarray(np.concatenate([a,label],axis=1)).resize((768,384),Image.Resampling.NEAREST).save(work/f'rig_{view}.png')
  assert all(v==0 for v in diff.values()),diff
  assert not qa['edge_touching_frames'],qa
  for act,kind in [('idle','idle'),('walk','gait')]:
   od=work/f'gait_{act}'
   if od.exists():shutil.rmtree(od)
   eval_frames=work/'loop_input'/act;eval_frames.mkdir(parents=True,exist_ok=True)
   # Give the estimator three complete repetitions so its lag search includes 16f.
   # These are evaluation inputs only; the exported animation remains 16 frames.
   for i in range(N*3):Image.fromarray(frames[('F',act)][i%N]).save(eval_frames/f'{i:03d}.png')
   res=subprocess.run([str(PY),'-I',str(FORGE/'skills/video2dsprite/scripts/gait_loop.py'),'select','--frames-dir',str(eval_frames),'--fps','13','--output-dir',str(od),'--kind',kind,'--no-aids'],capture_output=True,text=True)
   report=od/'loop-report.json';qa[f'gait_{act}']=json.loads(report.read_text()).get('recommendation') if report.exists() else {'error':res.stderr[-500:]}
  # Verify the exported 16-frame sequence, not only gait_loop's suggested subsequence.
  qa['exported_loops']={}
  for view in sources:
   for act in ['idle','walk']:
    seq=frames[(view,act)]
    def delta(a,b):
     mask=(a[...,3]>0)|(b[...,3]>0)
     return float(np.abs(a.astype(float)-b.astype(float))[mask].mean())
    steps=[delta(seq[i],seq[(i+1)%N]) for i in range(N)]
    qa['exported_loops'][view+'/'+act]={'seam_delta':steps[-1],'max_internal_delta':max(steps[:-1]),'seam_within_internal_range':steps[-1]<=max(steps[:-1])+1e-6,'unique_frames':len({a.tobytes() for a in seq})}
  qa['foot_contact']='Walk uses anatomical stride_arc: opposite phases, one stance foot at dy=0 throughout; no whole-body translation.'
  sheets={};bounds=[]
  for act in ACTS:
   rows=[]
   for v,flip in SRC:
    seq=[a[:,::-1] if flip else a for a in frames[(v,act)]];rows.append(np.concatenate(seq,axis=1))
    if act=='idle':
     ys,xs=np.nonzero(seq[0][...,3]);bounds.extend([int(xs.min()),int(ys.min()),int(np.ptp(xs)+1),int(np.ptp(ys)+1)])
   sheet=np.concatenate(rows,axis=0);sheets[act]=sheet
   for suffix in ['', '_8dir']:Image.fromarray(sheet).save(out/f'{act}{suffix}.png')
   Image.fromarray(sheet[:CH]).save(out/f'{act}_S_strip.png')
  master=np.concatenate(list(sheets.values()),axis=0)
  for file in ['sheet.png','all_8dir_actions.png']:Image.fromarray(master).save(out/file)
  Image.fromarray(np.concatenate([sheets['idle'][i*CH:(i+1)*CH,:CW] for i in range(8)],axis=1)).save(form/'8dir_stills.png')
  preview=Image.new('RGB',(CW*8,CH*8),(19,25,34))
  for di in range(8):
   for ai,act in enumerate(ACTS):
    for k,f in enumerate([0,7] if act!='hurt' else [1,5]):
     im=Image.fromarray(sheets[act][di*CH:(di+1)*CH,f*CW:(f+1)*CW]);preview.paste(im,((ai*2+k)*CW,di*CH),im)
  preview.save(out/'preview.png')
  actions=form/'actions';actions.mkdir(exist_ok=True);preview.save(actions/'preview.png')
  # Useful animated comparison, all four actions and both real source angles.
  anim=[]
  for f in range(N):
   im=Image.new('RGB',(CW*4,CH*2),(19,25,34));dr=ImageDraw.Draw(im)
   for vi,v in enumerate(['F','B']):
    for ai,act in enumerate(ACTS):
     a=Image.fromarray(frames[(v,act)][f]);im.paste(a,(ai*CW,vi*CH),a);dr.text((ai*CW+5,vi*CH+5),act,fill='white')
   anim.append(im)
  anim[0].save(out/'preview.gif',save_all=True,append_images=anim[1:],duration=90,loop=0,disposal=2)
  meta={'key':key,'dex':dex,'version':VERSION,'generator':Path(__file__).name,'art_source':'existing','source_art':['front.png','back.png'],'source_mapping':mapping,'direction_sources':dict(zip(DIRS,SRC)),'direction_limit':qa['direction_limit'],'cell_w':CW,'cell_h':CH,'foot_y':FY,'sprite_h':bounds[3],'frames':N,'actions':ACTS,'frame_time':AD,'directions':DIRS,'idle_bounds_S_to_SW':bounds,'palette_colors':len(cols),'off_palette_colors':0}
  (out/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));(out/'qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2))
  (work/'state.json').write_text(json.dumps({'hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'done':True}))
  if args.publish:
   pub=module('gen5_publish');pub.publish_dir8(key,{'dex':dex,'form':'gigantamax','variant':name+'-gmax','gen':'gen1','dir':rel,'sheetDir':'agent-sprite-forge','layout':'actions_8dir','nf':N,'cw':CW,'ch':CH,'w':CW,'h':bounds[3],'fy':FY,'an':[N]*4,'ad':AD,'raw':1,'src':'agent-sprite-forge','v':VERSION});pub.publish_bounds(key,bounds)
   sync=ROOT.parent/'pokemon_arena_2d'/rel/'agent-sprite-forge';sync.mkdir(parents=True,exist_ok=True)
   for p in out.iterdir():
    if p.is_file():shutil.copy2(p,sync/p.name)
   manifest=ROOT.parent/'pokemon_arena_2d/monsterSprite/forge-manifest.js'
   marker='// ORIGINAL-GMAX '+key
   text=manifest.read_text();start=text.find(marker+' BEGIN')
   if start>=0:
    end=text.index(marker+' END',start)+len(marker+' END');text=text[:start]+text[end:]
   entry={'bodyH':bounds[3],'bodyW':bounds[2],'anims':{}}
   for alias,act in [('Idle','idle'),('Walk','walk'),('Attack','attack'),('Strike','attack'),('Shoot','attack'),('SpAttack','attack'),('Charge','attack'),('Hurt','hurt')]:
    count=5 if alias=='Charge' else N
    entry['anims'][alias]={'fw':CW,'fh':CH,'anchorY':FY,'rows':8,'file':'../'+rel+'/agent-sprite-forge/'+act+'_8dir.png?v='+VERSION,'frames':count,'durations':[AD[ACTS.index(act)]*60]*count,'hit':7 if act=='attack' and alias!='Charge' else None}
   manifest.write_text(text.rstrip()+'\n'+marker+' BEGIN\nwindow.POKE_FORGE_ANIM.pokemon['+json.dumps(key)+'] = '+json.dumps(entry,separators=(',',':'))+';\n'+marker+' END\n')
  print(key,json.dumps(qa,ensure_ascii=False),flush=True)
if __name__=='__main__':main()

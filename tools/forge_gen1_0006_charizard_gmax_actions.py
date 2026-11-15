#!/usr/bin/env python3
"""Antigravity key art -> anatomical fire-wing kaiju actions, via agent-sprite-forge.

Reuses the Raichu X pipeline, not its anatomy. Source images are the recovered
Antigravity CLI art/out keys (S,W,NE,N + SW). Original front/back remain untouched.
Per-view joints: two flaming wings, tail, head, torso, arms and alternating legs.
Run with agent-sprite-forge/.venv/bin/python -I this_script.py [--force] [--no-sync].
"""
import argparse, hashlib, importlib.util, json, math, shutil, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parent.parent
REL='monsterSprite/gen1/0006_charizard/gigantamax/charizard-gmax'
FORM=ROOT/REL; ART=FORM/'art'; OUT=FORM/'agent-sprite-forge'; WORK=OUT/'_work'
FORGE=ROOT.parent/'agent-sprite-forge'; PY=FORGE/'.venv/bin/python'
SYNC=ROOT.parent/'pokemon_arena_2d'/REL/'agent-sprite-forge'
KEY='0006_charizard-gmax'; VERSION='charizard_gmax_agy_v1'
CW,CH,FY=224,144,126; N=16; UP=4
DIRS=['S','SE','E','NE','N','NW','W','SW']; ACTS=['idle','walk','attack','hurt']; AD=[.08,.062,.05,.055]
SRC={'S':('S',False),'SE':('SW',True),'E':('W',True),'NE':('NE',False),'N':('N',False),'NW':('NE',True),'W':('W',False),'SW':('SW',False)}
# Coordinates measured on each 688x384 key canvas. SW is scaled to this canvas.
# Later polygons own overlapping pixels. Torso owns the remaining opaque pixels.
RIG={
'S':dict(center=338,fwd=(0,1),mouth=[(320,112),(343,119),(358,112)],poly={
 'tail':[(420,260),(474,236),(490,185),(589,140),(596,235),(520,310),(435,340)],
 'wing_l':[(155,35),(277,35),(310,165),(287,190),(170,170)],
 'wing_r':[(370,165),(401,44),(603,38),(610,158),(398,194)],
 'leg_l':[(250,258),(307,280),(308,355),(285,383),(194,383),(215,339)],
 'leg_r':[(376,275),(430,258),(451,326),(489,384),(390,384),(377,344)],
 'head':[(296,0),(450,0),(385,135),(370,163),(312,159),(294,115)],
 'arm_l':[(274,180),(302,203),(272,240),(231,242),(178,220),(174,183),(222,178)],
 'arm_r':[(390,175),(433,182),(475,200),(477,250),(428,253),(403,222)]},
 piv={'tail':(435,300),'wing_l':(305,178),'wing_r':(380,178),'head':(338,150),'leg_l':(282,286),'leg_r':(402,286),'arm_l':(282,192),'arm_r':(407,191)}),
'W':dict(center=275,fwd=(-1,0),mouth=[(188,99),(141,112),(128,102)],poly={
 'tail':[(332,270),(403,302),(495,285),(529,213),(546,150),(592,145),(607,235),(567,315),(480,355),(344,345)],
 'wing_l':[(275,115),(297,30),(380,0),(500,0),(502,85),(437,150),(324,199)],
 'leg_l':[(187,261),(237,280),(252,340),(231,377),(163,375),(169,335)],
 'leg_r':[(273,247),(330,260),(351,328),(354,384),(264,384),(253,343)],
 'head':[(125,0),(351,0),(290,90),(246,125),(237,167),(205,169),(194,134),(122,120)],
 'arm_l':[(160,194),(216,191),(225,224),(193,249),(139,244),(138,213)],
 'arm_r':[(252,166),(294,180),(315,212),(312,255),(246,249),(205,227),(207,203)]},
 piv={'tail':(337,302),'wing_l':(293,176),'head':(222,158),'leg_l':(220,282),'leg_r':(292,276),'arm_l':(203,205),'arm_r':(280,186)}),
'NE':dict(center=350,fwd=(.9,-.45),mouth=[],poly={
 'tail':[(284,240),(315,285),(272,343),(168,352),(71,308),(33,236),(0,190),(0,131),(51,174),(73,254),(170,269),(253,254)],
 'wing_l':[(0,23),(145,27),(304,151),(302,199),(173,200),(44,165)],
 'wing_r':[(355,162),(423,77),(621,24),(655,102),(565,154),(392,205)],
 'leg_l':[(281,285),(350,292),(353,351),(345,381),(278,381),(273,345)],
 'leg_r':[(361,249),(411,258),(445,324),(480,384),(372,384),(355,330)],
 'head':[(270,0),(367,0),(421,79),(409,123),(365,138),(351,179),(300,171),(300,104)],
 'arm_l':[(251,183),(288,191),(295,227),(247,247),(214,224),(221,199)],
 'arm_r':[(403,184),(436,186),(476,200),(481,242),(429,253),(397,225)]},
 piv={'tail':(290,279),'wing_l':(305,180),'wing_r':(376,179),'head':(330,162),'leg_l':(314,298),'leg_r':(387,286),'arm_l':(277,201),'arm_r':(408,200)}),
'N':dict(center=340,fwd=(0,-1),mouth=[],poly={
 'wing_l':[(0,20),(154,18),(305,139),(317,191),(256,220),(133,190),(0,170)],
 'wing_r':[(370,148),(475,38),(663,20),(688,135),(591,180),(415,230),(370,205)],
 'leg_l':[(255,263),(315,282),(308,356),(290,384),(211,384),(237,329)],
 'leg_r':[(383,265),(440,266),(454,330),(469,384),(390,384),(378,333)],
 'tail':[(324,277),(371,267),(399,308),(440,335),(513,309),(546,254),(545,195),(581,166),(600,206),(588,285),(555,345),(490,381),(403,380),(345,343)],
 'head':[(290,0),(450,0),(384,103),(371,177),(310,174),(303,103)],
 'arm_l':[(243,177),(287,181),(292,225),(252,247),(202,221),(211,190)],
 'arm_r':[(402,181),(442,185),(477,208),(467,242),(424,249),(395,225)]},
 piv={'tail':(343,297),'wing_l':(302,178),'wing_r':(379,178),'head':(340,163),'leg_l':(280,287),'leg_r':(409,291),'arm_l':(267,191),'arm_r':(417,197)}),
'SW':dict(center=334,fwd=(-.9,.45),mouth=[(303,87),(284,99),(273,91)],poly={
 'tail':[(422,277),(457,258),(477,204),(552,145),(567,166),(548,222),(502,266),(498,308),(464,333),(422,322)],
 'wing_l':[(177,37),(245,50),(313,133),(284,165),(192,154),(184,111)],
 'wing_r':[(364,139),(390,50),(540,40),(563,110),(504,159),(405,179)],
 'leg_l':[(234,265),(287,274),(296,329),(269,360),(188,359),(204,329)],
 'leg_r':[(349,263),(408,258),(431,333),(479,384),(374,384),(355,342)],
 'head':[(275,0),(418,0),(367,74),(329,116),(324,163),(285,165),(286,110),(264,98)],
 'arm_l':[(256,168),(278,196),(257,227),(211,226),(182,209),(185,171),(218,166)],
 'arm_r':[(379,155),(417,172),(457,195),(453,240),(403,241),(391,205)]},
 piv={'tail':(433,293),'wing_l':(292,153),'wing_r':(382,158),'head':(308,148),'leg_l':(260,288),'leg_r':(385,289),'arm_l':(260,182),'arm_r':(397,175)})}

def module(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
mouth_open=module('mouth_open')
anatomical=module('forge_anatomical_actions')

def key_white(rgb):
 a=np.array(rgb.convert('RGB')); white=(a.min(2)>225)
 lab,_=ndi.label(white); border=np.unique(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]]));bg=np.isin(lab,border[border!=0])
 bg|=ndi.binary_dilation(bg,iterations=1)&(a.min(2)>210)
 return Image.fromarray(np.dstack([a,np.where(bg,0,255).astype('uint8')]))

def tool(args):
 p=subprocess.run([str(PY),'-I',str(FORGE/'skills/generate2dsprite/scripts/palette_tool.py'),*args],capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stdout+p.stderr)

def sources():
 im=Image.open(ART/'out/keys_2x2.png');w,h=im.size
 parts={v:im.crop(box) for v,box in {'S':(0,0,w//2,h//2),'W':(w//2,0,w,h//2),'NE':(0,h//2,w//2,h),'N':(w//2,h//2,w,h)}.items()}
 parts['SW']=Image.open(ART/'out/sw_single.png')
 keyed={v:key_white(i) for v,i in parts.items()}
 # Enclosed backdrop between jaw/neck and the flame wing (not belly/claws).
 holes={'SW':[(300,97),(336,97),(336,151),(302,151)],'NE':[(355,112),(404,112),(404,176),(355,176)],'W':[(239,119),(278,119),(278,175),(239,175)]}
 for v,poly in holes.items():
  a=np.array(keyed[v]);region=mask(poly,keyed[v].size);a[region&(a[...,:3].min(2)>175)&(np.ptp(a[...,:3].astype(int),axis=2)<50)]=0;keyed[v]=Image.fromarray(a)
 bbox=Image.open(FORM/'front.png').convert('RGBA').getbbox();height=bbox[3]-bbox[1]
 scale=height/(keyed['S'].getbbox()[3]-keyed['S'].getbbox()[1]); size=(round(w/2*scale),round(h/2*scale))
 raw=WORK/'stills_raw';raw.mkdir(parents=True,exist_ok=True)
 for v,im in keyed.items():
  sm=np.array(im.resize(size,Image.Resampling.BOX));sm[...,3]=np.where(sm[...,3]>=128,255,0);sm[sm[...,3]==0]=0;Image.fromarray(sm).save(raw/f'{v}.png')
 return size,height

def mask(poly,size):
 im=Image.new('L',size);ImageDraw.Draw(im).polygon([(x*size[0]/688,y*size[1]/384) for x,y in poly],fill=255);return np.array(im)>0

def make_rig(v,still):
 r=RIG[v];h,w=still.shape[:2];op=still[...,3]>0;names=['body',*r['poly']];lab=np.where(op,1,0)
 for i,n in enumerate(names[1:],2):lab[mask(r['poly'][n],(w,h))&op]=i
 masks={n:lab==i for i,n in enumerate(names,1)}
 order=[n for n in ['tail','wing_l','wing_r','leg_l','body','leg_r','head','arm_l','arm_r'] if n in names]
 if v in ('N','NE'):order=[n for n in ['leg_l','body','leg_r','head','arm_l','arm_r','wing_l','wing_r','tail'] if n in names]
 xoff=CW//2-round(r['center']*w/688);ys=np.nonzero(op.any(1))[0];yoff=FY-int(ys.max())
 cells={}; piv={};
 for ix,n in enumerate(order):
  m=masks[n];ly=np.zeros_like(still);ly[m]=still[m]
  above=np.logical_or.reduce([masks[z] for z in order[ix+1:]]) if ix+1<len(order) else np.zeros_like(m)
  # Canonical anatomical engine fills hidden joints with this part's own colors.
  ly=anatomical.build_part_layer(still,m,above,cushion_px=3)
  cell=np.zeros((CH,CW,4),np.uint8);cell[yoff:yoff+h,xoff:xoff+w]=ly;cells[n]=Image.fromarray(cell).resize((CW*UP,CH*UP),Image.Resampling.NEAREST)
  p=r['piv'].get(n,(r['center'],300));piv[n]=(p[0]*w/688+xoff,p[1]*h/384+yoff)
 base=np.zeros((CH,CW,4),np.uint8);base[yoff:yoff+h,xoff:xoff+w]=still
 mouth=[(x*w/688+xoff,y*h/384+yoff) for x,y in r['mouth']]
 return cells,piv,order,base,mouth

def wave(f,phase=0,cycles=1):return math.sin(2*math.pi*cycles*f/N+phase)-math.sin(phase)

def motion(v,n,act,f):
 r=RIG[v];fw,fy=r['fwd'];t=f/(N-1);s=wave(f);dx=dy=ang=0
 if act=='idle':
  breath=1.0*math.sin(math.pi*f/(N-1))**2
  if n in ('body','head','arm_l','arm_r'):dy=-breath
  if n.startswith('wing'):ang=(2.4 if n.endswith('l') else -2.4)*wave(f,-.4);dy=-breath
  if n=='tail':ang=1.7*wave(f,-.8)
 elif act=='walk':
  bob=.8*(1-math.cos(4*math.pi*f/N))
  if n.startswith('leg'):
   phase=0 if n.endswith('l') else math.pi
   q=math.sin(2*math.pi*f/N+phase);step=2.6*q
   dx=fw*step;dy=-3*max(0,q)
   # The support leg stays planted; translation is around the hip cushion.
   ang=0
  else:
   dy=-bob
   if n.startswith('arm'):dx=-fw*1.8*s*(1 if n.endswith('l') else -1)
   if n.startswith('wing'):ang=(3.2 if n.endswith('l') else -3.2)*wave(f,-.4)
   if n=='tail':ang=2.3*wave(f,-.8);dy=-bob*.45
 elif act=='attack':
  charge=[0,.25,.65,1,1,.3,0,0,0,0,0,0,0,0,0,0][f]
  hit=[0,0,0,0,0,.55,1,1,1,.85,.65,.4,.25,.12,0,0][f]
  if n in ('body','head'):dx=fw*(-2*charge+3*hit);dy=1.5*charge-hit
  if n.startswith('arm'):dx=fw*(-2*charge+4*hit);dy=-2*charge+hit;ang=(1 if n.endswith('l') else -1)*(3*charge-5*hit)
  if n.startswith('wing'):ang=(1 if n.endswith('l') else -1)*(6*charge-3*hit);dy=charge
  if n=='tail':ang=-2*charge+3*hit
  if n=='leg_r':dx=fw*2*hit;dy=fy*hit*.3
 elif act=='hurt':
  env=math.sin(math.pi*min(1,f/13)) if f<=13 else 0
  if not n.startswith('leg'):dx=-fw*3*env;dy=-env
  if n.startswith('wing'):ang=(1 if n.endswith('l') else -1)*7*env
  if n=='tail':ang=-3*env
 return round(dx),round(dy),ang

def compose(v,rig,act,f,palette):
 cells,piv,order,base,mouth=rig
 if act in ('attack','hurt') and f==15:return base.copy()
 out=Image.new('RGBA',(CW*UP,CH*UP))
 for n in order:
  dx,dy,ang=motion(v,n,act,f);im=cells[n]
  if abs(ang)>.2:im=im.rotate(-ang,resample=Image.Resampling.NEAREST,center=(piv[n][0]*UP+UP/2,piv[n][1]*UP+UP/2))
  out.alpha_composite(im,(dx*UP,dy*UP))
 a=np.array(out.resize((CW,CH),Image.Resampling.NEAREST));fw,fy=RIG[v]['fwd']
 if act=='attack' and 2<=f<=10:
  dx,dy,_=motion(v,'head',act,f);line=mouth_open.transform_points(mouth,dx=dx,dy=dy)
  if line:a=mouth_open.open_mouth(a,line,amount=2 if f>=5 else 1,deep=palette[0])
  if 5<=f<=10:
   # Fire is a directional mouth-origin plume. Back-facing plume is occluded by body.
   im=Image.fromarray(a);fx=Image.new('RGBA',(CW,CH));d=ImageDraw.Draw(fx)
   tip=line[-1] if line else (piv['head'][0]+dx,piv['head'][1]-12+dy)
   length=[12,22,28,28,23,16][f-5];normal=(-fy,fw)
   colors=sorted(palette,key=lambda c:c[0]-c[2]+.25*c[1],reverse=True)[:8]
   red=min(colors,key=lambda c:c[1]);yellow=max(colors,key=lambda c:c[1])
   for j in range(9):
    u=j/8;x=tip[0]+fw*length*u;y=tip[1]+fy*length*u
    rad=(2+4*u)*math.sin(math.pi*(j+1)/11)
    x+=normal[0]*math.sin(j*2+f)*u*2;y+=normal[1]*math.sin(j*2+f)*u*2
    d.ellipse((round(x-rad),round(y-rad),round(x+rad),round(y+rad)),fill=(*red,255))
    d.ellipse((round(x-rad*.45),round(y-rad*.45),round(x+rad*.45),round(y+rad*.45)),fill=(*yellow,255))
   if v in ('N','NE'):fx.alpha_composite(im);im=fx
   else:im.alpha_composite(fx)
   a=np.array(im)
 if act=='hurt' and f==1:a[a[...,3]>0,:3]=max(palette,key=lambda c:sum(c))
 a[a[...,3]==0]=0
 return a

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--force',action='store_true');ap.add_argument('--no-sync',action='store_true');ap.add_argument('--no-publish',action='store_true');args=ap.parse_args()
 WORK.mkdir(parents=True,exist_ok=True)
 h=hashlib.sha256(Path(__file__).read_bytes())
 for file in ('keys_2x2.png','sw_single.png'):h.update((ART/'out'/file).read_bytes())
 digest=h.hexdigest();state=WORK/'state.json';old=json.loads(state.read_text()) if state.exists() else {}
 if args.force or old.get('hash')!=digest:
  for d in ('stills_raw','palette','palette_lock','stills','frames','qa'):
   if (WORK/d).exists():shutil.rmtree(WORK/d)
 size,refheight=sources()
 if not (WORK/'stills').exists():
  tool(['build','--input',str(WORK/'stills_raw'),'--colors','32','--balance','--name','charizard-gmax','--output-dir',str(WORK/'palette')])
  tool(['lock','--palette',str(WORK/'palette/palette.json'),'--source',str(WORK/'stills_raw'),'--output-dir',str(WORK/'palette_lock')])
  tool(['apply','--palette',str(WORK/'palette_lock/palette.json'),'--input',str(WORK/'stills_raw'),'--orphans','1','--output-dir',str(WORK/'stills')])
 pj=json.loads((WORK/'palette_lock/palette.json').read_text());palette=[tuple(bytes.fromhex((c['hex'] if isinstance(c,dict) else c).lstrip('#'))[:3]) for c in pj['colors']];palette.sort(key=sum)
 still={v:np.array(Image.open(WORK/'stills'/f'{v}.png').convert('RGBA')) for v in RIG};rig={v:make_rig(v,im) for v,im in still.items()}
 frames={};diff={};qa={'off_palette_colors':0,'intermediate_alpha':0,'edge_touching_frames':[],'static_diff_pixels':{},'source_route':'Antigravity CLI (existing recovered key art)'}
 for v in RIG:
  for act in ACTS:
   path=WORK/'frames'/v/act;path.mkdir(parents=True,exist_ok=True)
   for f in range(N):
    p=path/f'{f:02d}.png'
    a=np.array(Image.open(p).convert('RGBA')) if p.exists() else compose(v,rig[v],act,f,palette)
    if not p.exists():Image.fromarray(a).save(p)
    frames[v,act,f]=a
   diff[f'{v}/{act}']=int(np.count_nonzero(np.any(frames[v,act,0]!=rig[v][3],axis=2)))
  print('rendered',v,flush=True)
 sheets={}
 for act in ACTS:
  sh=np.zeros((CH*8,CW*N,4),np.uint8)
  for di,d in enumerate(DIRS):
   v,flip=SRC[d]
   for f in range(N):sh[di*CH:(di+1)*CH,f*CW:(f+1)*CW]=frames[v,act,f][:,::-1] if flip else frames[v,act,f]
  sheets[act]=sh
 master=np.concatenate([sheets[a] for a in ACTS])
 used={tuple(c) for c in np.unique(master[master[...,3]>0,:3],axis=0)};qa['off_palette_colors']=len(used-set(palette));qa['intermediate_alpha']=int(np.count_nonzero((master[...,3]!=0)&(master[...,3]!=255)));qa['static_diff_pixels']=diff
 for (v,act,f),a in frames.items():
  op=a[...,3]>0
  if op[0].any() or op[-1].any() or op[:,0].any() or op[:,-1].any():qa['edge_touching_frames'].append(f'{v}/{act}/{f}')
 qa['manual_loop_checks']={}
 for v in RIG:
  for act in ('idle','walk'):
   seq=[frames[v,act,f] for f in range(N)]
   def delta(a,b):
    visible=(a[...,3]>0)|(b[...,3]>0)
    return float(np.abs(a.astype(float)-b.astype(float))[visible].mean())
   adjacent=[delta(seq[i],seq[i+1]) for i in range(N-1)]
   seam=delta(seq[-1],seq[0]);median=float(np.median(adjacent))
   endpoint=compose(v,rig[v],act,N,palette)
   qa['manual_loop_checks'][v+'/'+act]={'seam_delta':round(seam,4),'adjacent_median':round(median,4),'adjacent_max':round(max(adjacent),4),'seam_to_median':round(seam/median,4) if median else None,'virtual_frame_16_diff':int(np.count_nonzero(np.any(endpoint!=seq[0],axis=2)))}
 qa['walk_foot_contact']={v:{n:[motion(v,n,'walk',f)[1] for f in range(N)] for n in ('leg_l','leg_r')} for v in RIG}
 qa['loop_note']='16 unique samples of a periodic gait; virtual frame 16 equals frame 0. Last stored frame 15 is the preceding phase, not a duplicated endpoint. Foot offsets: 0 is support, negative is lift.'
 qa_dir=WORK/'qa';qa_dir.mkdir(exist_ok=True)
 for act,kind in [('idle','idle'),('walk','gait')]:
  d=qa_dir/f'gait_{act}_S'
  if d.exists():shutil.rmtree(d)
  p=subprocess.run([str(PY),'-I',str(FORGE/'skills/video2dsprite/scripts/gait_loop.py'),'select','--frames-dir',str(WORK/'frames/S'/act),'--fps',str(round(1/AD[ACTS.index(act)])),'--output-dir',str(d),'--kind',kind,'--no-aids'],capture_output=True,text=True)
  report=d/'loop-report.json';qa[f'gait_{act}']=json.loads(report.read_text()).get('recommendation') if report.exists() else {'error':p.stderr.strip()}
 (qa_dir/'qa.json').write_text(json.dumps(qa,indent=2))
 if qa['off_palette_colors'] or qa['intermediate_alpha'] or qa['edge_touching_frames'] or any(diff.values()):raise SystemExit('QC failed; see '+str(qa_dir/'qa.json'))
 for act,sh in sheets.items():
  for suffix in ('','_8dir'):Image.fromarray(sh).save(OUT/f'{act}{suffix}.png')
  Image.fromarray(sh[:CH]).save(OUT/f'{act}_S_strip.png')
 Image.fromarray(master).save(OUT/'sheet.png');Image.fromarray(master).save(OUT/'all_8dir_actions.png')
 pv=Image.new('RGBA',(CW*16,CH*8),(18,24,34,255))
 for ai,act in enumerate(ACTS):
  for j,f in enumerate([0,4,8,12]):pv.alpha_composite(Image.fromarray(sheets[act][:,f*CW:(f+1)*CW]),((ai*4+j)*CW,0))
 pv.save(OUT/'preview.png')
 idle0=[sheets['idle'][i*CH:(i+1)*CH,:CW] for i in range(8)];Image.fromarray(np.concatenate(idle0,axis=1)).save(FORM/'8dir_stills.png')
 bounds=[]
 for a in idle0:
  y,x=np.nonzero(a[...,3]);bounds.extend([int(x.min()),int(y.min()),int(x.max()-x.min()+1),int(y.max()-y.min()+1)])
 meta={'generator':Path(__file__).stem,'version':VERSION,'key':KEY,'dex':6,'art_source':'existing','route':'Antigravity CLI','source_art':['art/out/keys_2x2.png','art/out/sw_single.png'],'cell_w':CW,'cell_h':CH,'foot_y':FY,'sprite_h':bounds[3],'frames':N,'frame_time':AD,'directions':DIRS,'actions':ACTS,'direction_sources':SRC,'palette_colors':len(palette),'off_palette_colors':0,'idle_bounds_S_to_SW':bounds,'motion':'independent fire wings, head, arms, two alternating legs and counterbalancing tail; brace / exhale / recover','source_notes':['Recovered pre-existing Antigravity images; source edge contact retained in original art.'],'qa':'qa.json'}
 (OUT/'meta.json').write_text(json.dumps(meta,indent=2));shutil.copy2(qa_dir/'qa.json',OUT/'qa.json');shutil.copy2(WORK/'palette_lock/palette.json',OUT/'palette.json')
 state.write_text(json.dumps({'hash':digest,'version':VERSION,'complete':True}))
 if not args.no_sync:
  SYNC.mkdir(parents=True,exist_ok=True)
  for p in OUT.iterdir():
   if p.is_file():shutil.copy2(p,SYNC/p.name)
  shutil.copy2(FORM/'8dir_stills.png',SYNC.parent/'8dir_stills.png')
  manifest=ROOT.parent/'pokemon_arena_2d/monsterSprite/forge-manifest.js'
  if manifest.exists():
   import re
   text=manifest.read_text()
   text=re.sub(r"const geometry = \{[^}]+\};",f"const geometry = {{ fw: {CW}, fh: {CH}, anchorY: {FY}, rows: 8 }};",text)
   text=re.sub(r"const version = '[^']+';",f"const version = '{VERSION}';",text)
   text=re.sub(r'bodyH: \d+, bodyW: \d+',f'bodyH: {bounds[3]}, bodyW: {bounds[2]}',text)
   manifest.write_text(text)

 if not args.no_publish:
  pub=module('gen5_publish');pub.publish_dir8(KEY,{'dex':6,'form':'gigantamax','variant':'charizard-gmax','gen':'gen1','dir':REL,'sheetDir':'agent-sprite-forge','layout':'actions_8dir','nf':N,'cw':CW,'ch':CH,'w':CW,'h':bounds[3],'fy':FY,'an':[N]*4,'ad':AD,'raw':1,'src':'agent-sprite-forge','v':VERSION});pub.publish_bounds(KEY,bounds)
 status=ROOT/'monsterSprite/gen1/_forge_status'/f'{KEY}.json';status.parent.mkdir(exist_ok=True);status.write_text(json.dumps({'key':KEY,'status':'done','v':VERSION,'note':'Anatomical flame-wing kaiju; palette/alpha/edge/static QC passed. See qa.json for gait warnings.'}))
 print(json.dumps({'cell':[CW,CH],'foot_y':FY,'bounds_S':bounds[:4],'qa':qa},ensure_ascii=False))
if __name__=='__main__':main()

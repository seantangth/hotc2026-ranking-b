import csv, math
from collections import defaultdict
BASE="/Users/seantang/Desktop/Sean/The_Nexus/1_Projects/WHISPERS_2026_HyperSOT"
def load(p):
    d=defaultdict(dict)
    for r in csv.DictReader(open(p)):
        s,f=r['ID'].rsplit('_',1); d[s][int(f)]=[float(r['x']),float(r['y']),float(r['width']),float(r['height'])]
    return d
gt=load(f"{BASE}/1_data/raw/2026training.csv"); pr=load(f"{BASE}/5_outputs/e15_sam3_20260806/submission_val65.csv")
def cen(b): return (b[0]+b[2]/2,b[1]+b[3]/2)
def cle(a,b):
    (ax,ay),(bx,by)=cen(a),cen(b); return math.hypot(ax-bx,ay-by)
for seq in ['vis-droneshow2','rednir-droneshow2','rednir-drone2']:
    fr=sorted(gt[seq]); diag=math.hypot(gt[seq][fr[0]][2],gt[seq][fr[0]][3])
    C=[cle(gt[seq][f],pr[seq][f]) for f in fr if f in pr[seq]]
    S=sorted(C)
    # GT own speed
    gspd=[math.hypot(cen(gt[seq][fr[i]])[0]-cen(gt[seq][fr[i-1]])[0],
                     cen(gt[seq][fr[i]])[1]-cen(gt[seq][fr[i-1]])[1]) for i in range(1,len(fr))]
    # PRED own speed (jump magnitude of our own box)
    pf=[f for f in fr if f in pr[seq]]
    pspd=[math.hypot(cen(pr[seq][pf[i]])[0]-cen(pr[seq][pf[i-1]])[0],
                     cen(pr[seq][pf[i]])[1]-cen(pr[seq][pf[i-1]])[1]) for i in range(1,len(pf))]
    print(f"\n=== {seq} (n={len(C)}, GT diag {diag:.1f}px) ===")
    print(f"  CLE median={S[len(S)//2]:.1f}  p25={S[len(S)//4]:.1f}  p75={S[3*len(S)//4]:.1f}  max={S[-1]:.1f}")
    print(f"  frac frames CLE>2*diag = {100*sum(1 for c in C if c>2*diag)/len(C):.1f}%")
    print(f"  GT   speed: median={sorted(gspd)[len(gspd)//2]:.2f}px/f  max={max(gspd):.1f}")
    print(f"  PRED speed: median={sorted(pspd)[len(pspd)//2]:.2f}px/f  max={max(pspd):.1f}  (max = size of our box's teleport)")
    big=[(i+1,v) for i,v in enumerate(pspd) if v>3*diag]
    print(f"  frames where OUR box teleports >3*diag ({3*diag:.0f}px): {big[:6]}  (total {len(big)})")

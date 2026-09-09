import csv, math
from collections import defaultdict
BASE="/Users/seantang/Desktop/Sean/The_Nexus/1_Projects/WHISPERS_2026_HyperSOT"
def load(p):
    d=defaultdict(dict)
    for r in csv.DictReader(open(p)):
        seq,fi=r['ID'].rsplit('_',1); d[seq][int(fi)]=[float(r['x']),float(r['y']),float(r['width']),float(r['height'])]
    return d
gt=load(f"{BASE}/1_data/raw/2026training.csv"); pr=load(f"{BASE}/5_outputs/e15_sam3_20260806/submission_val65.csv")
def cen(b): return (b[0]+b[2]/2, b[1]+b[3]/2)
def cle(a,b):
    ax,ay=cen(a); bx,by=cen(b); return math.hypot(ax-bx,ay-by)

for seq in ['vis-droneshow2','rednir-droneshow2','rednir-drone2','rednir-droneshow4','vis-drone4']:
    fr=sorted(gt[seq]); w,h=gt[seq][fr[0]][2],gt[seq][fr[0]][3]; diag=math.hypot(w,h)
    ser=[(k,f,cle(gt[seq][f],pr[seq][f])) for k,f in enumerate(fr) if f in pr[seq]]
    C=[x[2] for x in ser]
    # true switch = largest single-frame jump in CLE
    jumps=sorted(((C[i]-C[i-1], i) for i in range(1,len(C))), reverse=True)
    dj,si=jumps[0]
    print(f"\n=== {seq}: {len(ser)} frames, GT {w:.0f}x{h:.0f} diag={diag:.1f}px ===")
    print(f"  biggest jump +{dj:.1f}px at local frame {si} ({100*si/len(C):.0f}%), CLE {C[si-1]:.1f} -> {C[si]:.1f}")
    print(f"  top-3 jumps: "+", ".join(f"f{i}:+{d:.0f}px" for d,i in jumps[:3]))
    # was pre-switch memory clean? use "on target" = CLE < 2*diag
    ok=2*diag
    for span,label in [(6,'r=1 maskmem  (prev 6)'),(15,'r=1 obj_ptr  (prev 15)'),
                       (26,'r=5 maskmem  (prev ~26)'),(71,'r=5 obj_ptr  (prev ~71)')]:
        wnd=C[max(0,si-span):si]
        frac=sum(1 for c in wnd if c<ok)/len(wnd)
        print(f"  {label:24s} medCLE={sorted(wnd)[len(wnd)//2]:7.1f}px  on-target={100*frac:5.1f}%  "
              f"{'ALL-CLEAN' if frac==1.0 else 'partly contaminated'}")
    print("  trace around switch: "+" ".join(f"{C[j]:.0f}" for j in range(max(0,si-10),min(len(C),si+5))))

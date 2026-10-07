
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from train_rescue_pipeline import make_model

EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

def arr(p): return np.array(Image.open(p))

def norm(x):
    x=x.astype(np.float32)
    lo,hi=np.percentile(x,1),np.percentile(x,99)
    return np.clip((x-lo)/(hi-lo+1e-6),0,1)

def pairs(root, split):
    imd=Path(root)/"images"/split; mskd=Path(root)/"masks"/split
    ims={p.stem:p for p in imd.iterdir() if p.suffix.lower() in EXTS}
    ms={p.stem:p for p in mskd.iterdir() if p.suffix.lower() in EXTS}
    return [(ims[k],ms[k]) for k in sorted(set(ims)&set(ms))]

@torch.inference_mode()
def infer(model, img, device):
    x=torch.from_numpy(norm(img)[None,None]).float().to(device)
    return torch.softmax(model(x),1)[0,1].cpu().numpy()

def md(prob,gt,t=.5):
    pr=prob>=t
    tp=np.logical_and(pr,gt).sum()
    fp=np.logical_and(pr,~gt).sum()
    fn=np.logical_and(~pr,gt).sum()
    return tp/(tp+fp+fn+1e-8), 2*tp/(2*tp+fp+fn+1e-8)

def overlay(img, pred):
    b=np.stack([norm(img)]*3,-1)
    o=b.copy()
    o[pred]=0.55*o[pred]+0.45*np.array([1.,.1,.1])
    return np.clip(o,0,1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default=".")
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--split",default="val")
    ap.add_argument("--examples",type=int,default=6)
    ap.add_argument("--out",default="artifacts/paper_figures")
    args=ap.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=make_model().to(device)
    ck=torch.load(args.checkpoint,map_location=device)
    model.load_state_dict(ck["model"] if isinstance(ck,dict) and "model" in ck else ck)
    model.eval()

    ps=pairs(Path(args.project)/"external"/"refined_sos",args.split)
    if not ps: raise SystemExit("No matching image/mask pairs found.")

    res=[]
    for ip,mp in ps:
        im=arr(ip)
        if im.ndim==3: im=im[...,0]
        gt=arr(mp)
        if gt.ndim==3: gt=gt[...,0]
        gt=gt>0
        prob=infer(model,im,device)
        i,d=md(prob,gt)
        res.append((i,d,ip,im,gt,prob))

    res.sort(key=lambda z:z[0],reverse=True)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    meta=[]

    for n,(iou,dice,ip,im,gt,prob) in enumerate(res[:args.examples],1):
        pred=prob>=0.5
        fig,ax=plt.subplots(1,5,figsize=(20,4))
        ax[0].imshow(norm(im),cmap="gray"); ax[0].set_title("SAR Input")
        ax[1].imshow(gt,cmap="gray"); ax[1].set_title("Ground Truth")
        ax[2].imshow(prob,cmap="viridis",vmin=0,vmax=1); ax[2].set_title("Oil Probability")
        ax[3].imshow(pred,cmap="gray"); ax[3].set_title("Predicted Mask")
        ax[4].imshow(overlay(im,pred)); ax[4].set_title(f"Overlay | IoU {iou*100:.1f}%")
        for a in ax:a.axis("off")
        plt.tight_layout()
        fp=out/f"refined_sos_val_example_{n:02d}.png"
        fig.savefig(fp,dpi=300,bbox_inches="tight")
        plt.close(fig)
        meta.append({"rank":n,"filename":ip.name,"iou":float(iou),"dice":float(dice),"figure":str(fp)})

    (out/"figure_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("Saved:",out.resolve())
    for m in meta: print(f"{m['rank']}: {m['filename']} IoU={m['iou']:.4f} Dice={m['dice']:.4f}")

if __name__=="__main__":
    main()

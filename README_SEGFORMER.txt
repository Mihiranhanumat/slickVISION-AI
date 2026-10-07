SLICKVISION-AI SEGFORMER UPGRADE

1) Install:
   pip install transformers

2) Check:
   python scripts/check_segformer_setup.py

3) Screening configuration:
   SegFormer MiT-B2, binary classes, 256x256, CE+Dice, AdamW,
   augmentation enabled, 15-epoch screening, batch size 4.

4) IMPORTANT:
   Keep the existing E3 checkpoint untouched.
   Select architecture/epoch/loss/threshold using VALIDATION ONLY.
   Do not tune on the test split.

5) After the setup check passes, we will add/run the trainer against
   your exact existing dataset loader. I have deliberately not included
   a second copy of your dataset pipeline because your current repo
   already has a working loader and we should not risk breaking it.

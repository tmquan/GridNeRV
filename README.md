# GridNeRV
VoxelNeRV

```bash
conda create -n py310 python=3.10
conda activate py310
pip install -U monai[all]
pip install -U diffusers
pip install -U lightning
pip install -U transformers
pip install git+https://github.com/tatp22/multidim-positional-encoding 
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
# conda install pytorch3d -c pytorch3d
# conda install pytorch-lightning
# conda install monai
python setup.py
```

```
docker run -it nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 bash
```

```bash
env CUDA_VISIBLE_DEVICES='4,5,6,7' python main_gridnerv1.py --accelerator='cuda' --devices=4 --batch_size=1 --lr=1e-4 --logsdir=/logs_gridnerv1_sh2_pe8 --datadir=data --train_samples=2000 --val_samples=400 --n_pts_per_ray=256 --shape=256 --alpha=1 --theta=1 --gamma=10 --omega=1  --sh=2 --pe=8 --amp
```

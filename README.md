# TemporalOcclusion

```bash
conda create -n py3d python=3.9 -y
conda activate py3d
pip install "numpy<2.0"
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html
pip install polyscope
```
# ComfyUI-MuseTalk_FSH
the comfyui custom node of [MuseTalk](https://github.com/TMElyralab/MuseTalk.git) to make audio driven videos!
<div>
  <figure>
  <img alt='webpage' src="web.png?raw=true" width="600px"/>
  <figure>
</div>

## How to use
make sure `ffmpeg` is worked in your commandline
for Linux
```
apt update
apt install ffmpeg
```
for Windows,you can install `ffmpeg` by [WingetUI](https://github.com/marticliment/WingetUI) automatically

then!
```
git clone https://github.com/AIFSH/ComfyUI-MuseTalk_FSH.git
cd ComfyUI-MuseTalk_FSH
pip install -r requirements.txt
```
### mmlab packages
```bash
pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0" 
```

### Download weights
You can download weights manually as follows:

1. Download our trained [weights](https://huggingface.co/TMElyralab/MuseTalk).

2. Download the weights of other components:
   - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
   - [whisper](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt)
   - [dwpose](https://huggingface.co/yzd-v/DWPose/tree/main)
   - [face-parse-bisent](https://github.com/zllrunning/face-parsing.PyTorch)
   - [resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth)

或者下载[MuseTalk.zip](https://pan.quark.cn/s/6efa251a2609)，
解压后把子文件夹放入`ComfyUI-MuseTalk_FSH/models/`目录

Finally, these weights should be organized in `models` as follows:
```
ComfyUI-MuseTalk_FSH/models/
├── musetalk
│   └── musetalk.json
│   └── pytorch_model.bin
├── dwpose
│   └── dw-ll_ucoco_384.pth
├── face-parse-bisent
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
├── sd-vae-ft-mse
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── whisper
    └── tiny.pt
```

## Tutorial
- [Demo on 3060 12GB](https://www.bilibili.com/video/BV1St421w7Qn)
- [Demo on 4090 24GB](https://www.bilibili.com/video/BV13T42117uM/)


## WeChat Group && Donate
<div>
  <figure>
  <img alt='Wechat' src="wechat.jpg?raw=true" width="300px"/>
  <img alt='donate' src="donate.jpg?raw=true" width="300px"/>
  <figure>
</div>
    
## Thanks
- [MuseTalk](https://github.com/TMElyralab/MuseTalk.git) 

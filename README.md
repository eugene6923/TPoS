<!-- <style>
red { color: red }
yellow { color: yellow }
</style> -->

# The Power of Sound(TPoS): Audio Reactive Video Generation with Stable Diffusion

This is the official implementation of the paper: "The Power of Sound(TPoS): Audio Reactive Video Generation with Stable Diffusion (ICCV 2023)".

<!-- [[Paper]()] [[Project]()] -->
[[Paper](https://arxiv.org/abs/2309.04509)] [[Project Page](https://ku-vai.github.io/TPoS/)]


## Usage
Following code is based on Stable Diffusion. So for more details, you can visit [link](https://github.com/CompVis/stable-diffusion). Get the checkpoints for Stable Diffusion.

```
gh repo clone ku-vai/TPoS
```



### Train Audio Encoder
 You can find `audio_encoder/train.py` to train the Audio Encoder. You need two datasets ([Landscape](https://kuai-lab.github.io/eccv2022sound/) and [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)). This code is based on [Sound-guided Semantic Image Manipulation (CVPR2022)](https://github.com/kuai-lab/sound-guided-semantic-image-manipulation).
    
You can use the following codes for training. 


```
cd audio_encoder
python train_audio_encoder.py
```

Or you can simply download our pretrained weights from following link: [link](https://drive.google.com/drive/folders/11kDpSAp6wKyDU13rVT66dB0H2vJwXk5D?usp=drive_link). Locate downloaded weights in `pretrained_models`.


### Video Generation with Sound
When you want to test your model with image dataset, you can easily run the code with `bash inference.sh`. You can change the audio and text prompt.


## Citation
```
@inproceedings{jeong2023power,
  title={The power of sound (tpos): Audio reactive video generation with stable diffusion},
  author={Jeong, Yujin and Ryoo, Wonjeong and Lee, Seunghyun and Seo, Dabin and Byeon, Wonmin and Kim, Sangpil and Kim, Jinkyu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7822--7832},
  year={2023}
}
```

"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
import sys
sys.path.append("./")

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.sound.util import Audio_generator


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=1,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="../sound/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=2
    )
    parser.add_argument(
        "--interpolate",
        type=int,
        default=None
    )
    parser.add_argument(
        "--audio",
        type=str,
        default= None,
        help="if enabled, uses the audio directory",
    )
    parser.add_argument(
        "--audio_num",
        type=float,
        default= 20
    )
    parser.add_argument(
        "--audio_time",
        type=int,
        default= None
    )
    parser.add_argument(
        "--outname",
        type=str,
        default= "samples",
    )
    parser.add_argument(
        "--skip_first",
        action='store_true',
    )
    parser.add_argument(
        "--text_condition",
        type=str,
        default = ""
    )
    parser.add_argument(
        "--only_audio",
        action='store_true',
    )
    parser.add_argument(
        "--start_token",
        action='store_true',
    )
    parser.add_argument(
        "--pt_file_name",
        type=str,
        default = None
    )
    parser.add_argument(
        "--se_value",
        type=float,
        default = 4
    )
    parser.add_argument(
        "--threshold_value",
        type=float,
        default = 0.85
    )
    parser.add_argument(
        "--param_t",
        type=int,
        default = 800
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default = 0
    )


    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    # if opt.audio is not None:
    #     opt.outname = os.path.splitext(os.path.split(opt.audio)[1])[0] + "_" +  str(float(opt.audio_num))
    # sample_path = os.path.join(outpath, opt.outname)
    # os.makedirs(sample_path, exist_ok=True)
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))
    
    condition_list = []
    audio_time_list = []
    if opt.audio != None:
        for i in range(opt.audio_time) :
            audio_time_list.append(i)
            condition_list.append(opt.audio)

    elif opt.audio == None and opt.text_condition != None:
        audio_time_list.append(None)
        condition_list.append(opt.text_condition)

    if opt.skip_first: using_condition_list = []
    else: using_condition_list = [model.get_learned_conditioning(batch_size * [prompt], [None,0,None])]
    
    if os.path.exists(f"./{opt.pt_file_name}.pt"):
        os.remove(f"./{opt.pt_file_name}.pt")
        os.remove(f"./{opt.pt_file_name}_volume.pt")

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                start_token = opt.start_token
                for prompts in tqdm(data, desc="data"):
                    if start_token:
                        using_condition_list = []
                        start_token = False
                    file_name = opt.pt_file_name
                    if opt.audio != None:
                        with Audio_generator(opt.audio, opt.audio_time, opt.pt_file_name):
                            for i, condition in enumerate(condition_list):
                                audio=[condition,opt.audio_num,audio_time_list[i]]
                                # audio =[None,0,None]
                                using_condition_list.append(model.get_learned_conditioning(batch_size * [opt.text_condition], audio, file_name))
                    else:
                        if opt.text_condition != "":
                            for i, condition in enumerate(condition_list):
                                audio = [None,0,None]
                                using_condition_list.append(model.get_learned_conditioning(batch_size * [opt.text_condition], audio, file_name))

    def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > DOT_THRESHOLD: v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1
        return torch.from_numpy(v2)

    interpolated_list = []
    if opt.interpolate is None or opt.interpolate==1:
        interpolated_list = using_condition_list
    else:
        interpolation_weight = np.linspace(0.0, 1.0, opt.interpolate)
        interpolated_list = []
        for channel_idx in range(len(using_condition_list)-1):
            for interpolation_idx in range(opt.interpolate):
                if interpolation_idx == 0 and channel_idx != 0:
                    continue
                interpolated_list.append(slerp(interpolation_weight[interpolation_idx], using_condition_list[channel_idx].cpu().detach().numpy(), using_condition_list[channel_idx+1].cpu().detach().numpy()))
        interpolated_list = torch.cat(interpolated_list).to(device).unsqueeze(1)



    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    assert os.path.isfile(opt.init_img)
    init_image = load_img(opt.init_img).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")
    

    for n in trange(opt.n_iter, desc="Sampling"):
        # seed_everything(opt.seed+n)
        # print(f"seed : {opt.seed+n}")
        if opt.audio is not None:
            opt.outname = os.path.splitext(os.path.split(opt.audio)[1])[0] + "_" +  str(float(opt.audio_num))
        sample_path = os.path.join(outpath, opt.outname)
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1
        for frames_idx, interpolated_condition in enumerate(interpolated_list): 
            seed_everything(opt.seed+n)
            precision_scope = autocast if opt.precision == "autocast" else nullcontext
            with torch.no_grad():
                # os.makedirs(opt.outdir, exist_ok=True)
                # outpath = opt.outdir
                with precision_scope("cuda"):
                    with model.ema_scope():
                        tic = time.time()
                        all_samples = list()
                        # for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            default_uc = model.get_learned_conditioning(batch_size * [""], [None, 0, None])
                            if opt.skip_first:
                                uc = [interpolated_condition]
                                if opt.audio != None and opt.text_condition!="":
                                    uc.append(model.get_learned_conditioning([opt.text_condition], [None,0,None]))
                            else:
                                if frames_idx != 0:
                                    uc = [interpolated_condition]
                                    if opt.audio != None and opt.text_condition!="": uc.append(model.get_learned_conditioning([opt.text_condition], [None,0,None]))
                                else: uc = []
                        # if opt.scale != 1.0:
                        #     uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts, [None,0,None])

                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                        # decode it
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                unconditional_conditioning=uc, default_uc =default_uc, param_t=opt.param_t, se_value=opt.se_value, threshold_value=opt.threshold_value)

                        x_samples = model.decode_first_stage(samples[0])
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1
                                print("Done!")
                        all_samples.append(x_samples)

                        if not opt.skip_grid:
                            # additionally, save as grid
                            grid = torch.stack(all_samples, 0)
                            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                            grid = make_grid(grid, nrow=n_rows)

                            # to image
                            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                            Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                            grid_count += 1

                        toc = time.time()


        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
            f" \nEnjoy.")
        # input()


if __name__ == "__main__":
    main()

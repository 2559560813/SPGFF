import json
import os.path as osp
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",  
}

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
        
    model = clip.build_model(state_dict or model.state_dict())

    return model.float()



class VisionPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        width = clip_model.visual.width
        self.width = width
        self.prompt_till_layer_visual = cfg.TRAINER.VPT.PROMPT_DEPTH_VISION  # max=12, but will create 11 such shared prompts

        assert self.prompt_till_layer_visual >= 1
        self.n_ctx_visual = cfg.TRAINER.VPT.N_CTX_VISION  # hyperparameter

        # These below, related to the shallow prompts        
        self.VP_shallow = nn.Parameter(torch.empty(self.n_ctx_visual, width))
        nn.init.normal_(self.VP_shallow, std=0.02)
        # These below, related to the deep prompts
        self.VP_deep = nn.ParameterList([nn.Parameter(torch.empty(self.n_ctx_visual, width))for _ in range(self.prompt_till_layer_visual - 1)])
        for VP_deep_i in self.VP_deep:
            nn.init.normal_(VP_deep_i, std=0.02)
        



class CustomCLIP(nn.Module):
    def __init__(self, cfg, dm, classnames, clip_model):
        super().__init__()
        self.dtype = clip_model.dtype
        self.dm = dm
        self.dataset = cfg.DATASET.NAME
        self.cfg = cfg

        # Get hard text features
        self.n_cls = len(classnames)
        self.text_encoder = clip_model.cuda().encode_text
        self.hard_text_features = self.get_hard_text_features(self.n_cls)
        
        # Get image encoder
        self.image_encoder = clip_model.visual
        self.vision_prompt_learner = VisionPromptLearner(cfg, clip_model)
        
        self.logit_scale = clip_model.logit_scale


    # Get text features of all classes
    def get_hard_text_features(self, n_cls):
        text_features = None
        for label in range(n_cls):
            classname = self.dm.lab2cname[label].replace("_", " ") 
            temp = CUSTOM_TEMPLATES[self.dataset]
            hard_prompt = temp.format(classname)  
            tokenized_hard_prompt = clip.tokenize(hard_prompt).cuda()
            with torch.no_grad():
                text_feature = self.text_encoder(tokenized_hard_prompt)

            if text_features == None:
                text_features = text_feature
            else:
                text_features = torch.concat([text_features,text_feature],dim=0)
        return text_features.cuda()


    @staticmethod
    def norm(x):
            return x / x.norm(dim=-1, keepdim=True)
        
    def forward(self, image, label=None, training=False):
        prompt_image_feature = self.image_encoder(image.type(self.dtype), self.vision_prompt_learner.VP_shallow, self.vision_prompt_learner.VP_deep)[0]
        
        hard_text_features = self.hard_text_features
        
        norm_prompt_image_feature = self.norm(prompt_image_feature)
        norm_hard_text_features = self.norm(hard_text_features)
        logits = self.logit_scale.exp() * norm_prompt_image_feature @ norm_hard_text_features.t()

        if training:
            class_loss = F.cross_entropy(logits, label) 
            return class_loss
        
        return logits


@TRAINER_REGISTRY.register()
class VPT(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.VPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.VPT.PREC == "fp32" or cfg.TRAINER.VPT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, self.dm, classnames, clip_model)

        print("Turning off gradients in only the image encoder")

        for name, param in self.model.named_parameters():
            if (("vision_prompt_learner" not in name)):
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.vision_prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.vision_prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.VPT.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)



    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.VPT.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label, training=True)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

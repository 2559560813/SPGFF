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

    return model


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm_text = nn.LayerNorm(dim)
        self.norm_img = nn.LayerNorm(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, query, kv):
        query = self.q(query)
        k, v = kv, kv

        attn = (query @ k.transpose(1,2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        query = attn @ v
        query = self.proj(query)
        query = self.proj_drop(query)

        return query



class CrossAttnBlock(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., norm_layer=nn.LayerNorm,):
        super(CrossAttnBlock, self).__init__()
        self.self_attn_layer = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.norm2_q = norm_layer(dim)
        self.norm2_kv = norm_layer(dim)

    def forward(self, query, text_features):
        query = query + self.self_attn_layer(self.norm1(query))
        query = query + self.cross_attn(self.norm2_q(query), self.norm2_kv(text_features))
        return query


class TextGuidedDeepPrompt(nn.Module):
    def __init__(self, cfg, clip_model, text_dim, hidden_dim=32):
        super().__init__()
        width = clip_model.visual.width
        self.width = width
        self.prompt_till_layer_visual = cfg.TRAINER.SPGFF.PROMPT_DEPTH_VISION  # max=12, but will create 11 such shared prompts

        assert self.prompt_till_layer_visual >= 1
        self.n_ctx_visual = cfg.TRAINER.SPGFF.N_CTX_VISION  # hyperparameter

        self.mlp = nn.Sequential(
            nn.Linear(2 * text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.prompt_till_layer_visual * self.n_ctx_visual * self.width)
        )
        self.bonder = CrossAttnBlock(text_dim)
        if cfg.TRAINER.SPGFF.PREC == 'fp16':
            self.mlp.half()
            self.bonder.half()

    def forward(self, image_feat, text_features):
        attn_weights = torch.matmul(image_feat, text_features.T) 
        attn_weights = torch.softmax(attn_weights, dim=-1) 
        T_weighted = torch.sum(attn_weights.unsqueeze(-1) * text_features, dim=1) 
        x_t = torch.cat([image_feat, T_weighted], dim=-1) 
        P_x = self.mlp(x_t).view(-1, self.prompt_till_layer_visual, self.n_ctx_visual, self.width)
        return P_x



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
        self.gpt_text_features = self.get_gpt_text_features(self.n_cls)
        self.hard_text_features = self.get_hard_text_features(self.n_cls)
        text_dim = self.gpt_text_features.shape[1]
        
        # Get image encoder
        self.image_encoder = clip_model.visual
        self.vision_prompt_learner = TextGuidedDeepPrompt(cfg, clip_model, text_dim)
        
        self.logit_scale = clip_model.logit_scale
        self.alpha = cfg.TRAINER.SPGFF.ALPHA
        self.lamda = cfg.TRAINER.SPGFF.LAMDA

        self.class_text = cfg.TRAINER.SPGFF.CLASS_TEXT
        self.fusion_text = cfg.TRAINER.SPGFF.FUSION_TEXT

    # Get GPT3 text feature of someone class
    def get_gpt_text_features(self, n_cls):
        gpt3_text_features = []
        for label in range(n_cls):
            # Get the classname of label
            classname = self.dm.lab2cname[label].replace("_", " ") 
            # Read the pre-stored texts generated by GPT3 for each class
            current_dir = os.path.dirname(os.path.abspath(__file__))
            descriptions_file = osp.join(current_dir, "../gpt_file", self.dataset+'.json')
            with open(descriptions_file) as f:
                gpt3_descriptions = json.load(f)          
            texts = gpt3_descriptions[classname] 
            # Tokenize the texts    
            tokenized_gpt3_prompts = clip.tokenize(texts).cuda() # (5, 77)
            with torch.no_grad():
                gpt3_text_features_i = self.text_encoder(tokenized_gpt3_prompts)  # (5, 512)
            gpt3_text_feature_i = gpt3_text_features_i.mean(dim=0, keepdim=True) # (1, 512)

            gpt3_text_features.append(gpt3_text_feature_i)
             
        gpt3_text_features = torch.stack(gpt3_text_features, dim=0).squeeze(1)
        return gpt3_text_features


    # Get Tip text features of all classes
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
        clip_image_feature = self.image_encoder(image.type(self.dtype))[0]
        fusion_text_features = self.hard_text_features  if self.fusion_text == 'hard' else self.gpt_text_features

        prompts = self.vision_prompt_learner(clip_image_feature, fusion_text_features).squeeze(0)
        shallow_prompt = prompts[0, :, :]
        deep_prompts = prompts[1:, :, :]
        prompt_image_feature, prompt_patch_tokens = self.image_encoder(image.type(self.dtype), shallow_prompt, deep_prompts)

        text_guide_image_feature = self.vision_prompt_learner.bonder(prompt_patch_tokens, fusion_text_features.unsqueeze(0))
        
        text_guide_image_feature = text_guide_image_feature.mean(dim=1)
        fusion_image_feature = prompt_image_feature + self.alpha * text_guide_image_feature 
        
        norm_fusion_image_feature = self.norm(fusion_image_feature)
        class_text_features = self.hard_text_features  if self.class_text == 'hard' else self.gpt_text_features
        norm_class_text_features = self.norm(class_text_features)
        logits = self.logit_scale.exp() * norm_fusion_image_feature @ norm_class_text_features.t()

        if training:
            score = F.cosine_similarity(fusion_image_feature, clip_image_feature)
            kd_loss = 1.0-score

            class_loss = F.cross_entropy(logits, label)
            loss = class_loss + self.lamda * kd_loss 
            return loss, class_loss, kd_loss
        
        return logits


@TRAINER_REGISTRY.register()
class SPGFF(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.SPGFF.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.SPGFF.PREC == "fp32" or cfg.TRAINER.SPGFF.PREC == "amp":
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

        self.scaler = GradScaler() if cfg.TRAINER.SPGFF.PREC == "amp" else None

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

        prec = self.cfg.TRAINER.SPGFF.PREC
        if prec == "amp":
            with autocast():
                loss, class_loss, kd_loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss, class_loss, kd_loss = model(image, label, training=True)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item(),
                        "class_loss": class_loss.item(),
                        "kd_loss": kd_loss.item()}

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

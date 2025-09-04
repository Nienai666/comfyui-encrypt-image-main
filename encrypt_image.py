
import base64
import io
import json
import os
from pathlib import Path
from urllib.parse import unquote
from .core.core import get_sha256,dencrypt_image,dencrypt_image_v2,encrypt_image_v2
from PIL import PngImagePlugin,_util,ImagePalette
from PIL import Image as PILImage
from io import BytesIO
from typing import Optional
import sys
import folder_paths
from comfy.cli_args import args

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import numpy as np

_password = '123qwe'

            
if PILImage.Image.__name__ != 'EncryptedImage':
    
    super_open = PILImage.open
    
    class EncryptedImage(PILImage.Image):
        __name__ = "EncryptedImage"
        @staticmethod
        def from_image(image:PILImage.Image):
            image = image.copy()
            img = EncryptedImage()
            img.im = image.im
            img._mode = image.im.mode
            if image.im.mode:
                try:
                    img.mode = image.im.mode
                except Exception as e:
                    ''
            img._size = image.size
            img.format = image.format
            if image.mode in ("P", "PA"):
                if image.palette:
                    img.palette = image.palette.copy()
                else:
                    img.palette = ImagePalette.ImagePalette()
            img.info = image.info.copy()
            return img
            
        def save(self, fp, format=None, **params):
            filename = ""
            if isinstance(fp, Path):
                filename = str(fp)
            elif _util.is_path(fp):
                filename = fp
            elif fp == sys.stdout:
                try:
                    fp = sys.stdout.buffer
                except AttributeError:
                    pass
            if not filename and hasattr(fp, "name") and _util.is_path(fp.name):
                # only set the name for metadata purposes
                filename = fp.name
            
            if not filename or not _password:
                # 如果没有密码或不保存到硬盘，直接保存
                super().save(fp, format = format, **params)
                return
            
            if 'Encrypt' in self.info and (self.info['Encrypt'] == 'pixel_shuffle' or self.info['Encrypt'] == 'pixel_shuffle_2'):
                super().save(fp, format = format, **params)
                return
            
            encrypt_image_v2(self, get_sha256(_password))
            self.format = PngImagePlugin.PngImageFile.format
            pnginfo = params.get('pnginfo', PngImagePlugin.PngInfo())
            if not pnginfo:
                pnginfo = PngImagePlugin.PngInfo()
                for key in (self.info or {}).keys():
                    if self.info[key]:
                        pnginfo.add_text(key,str(self.info[key]))
            pnginfo.add_text('Encrypt', 'pixel_shuffle_2')
            pnginfo.add_text('EncryptPwdSha', get_sha256(f'{get_sha256(_password)}Encrypt'))
            params.update(pnginfo=pnginfo)
            super().save(fp, format=self.format, **params)
            # 保存到文件后解密内存内的图片，让直接在内存内使用时图片正常
            dencrypt_image_v2(self, get_sha256(_password)) 
            
    def open(fp,*args, **kwargs):
        image = super_open(fp,*args, **kwargs)
        if _password and image.format.lower() == PngImagePlugin.PngImageFile.format.lower():
            pnginfo = image.info or {}
            if 'Encrypt' in pnginfo and pnginfo["Encrypt"] == 'pixel_shuffle':
                dencrypt_image(image, get_sha256(_password))
                pnginfo["Encrypt"] = None
                image = EncryptedImage.from_image(image=image)
                return image
            if 'Encrypt' in pnginfo and pnginfo["Encrypt"] == 'pixel_shuffle_2':
                dencrypt_image_v2(image, get_sha256(_password))
                pnginfo["Encrypt"] = None
                image = EncryptedImage.from_image(image=image)
                return image
        return EncryptedImage.from_image(image=image)

    # if _password:
    PILImage.Image = EncryptedImage
    PILImage.open = open
    
    print('图片加密插件加载成功')

# 这是一个节点，用于设置密码，即使不设置，也有默认密码 123qwe
class EncryptImage:
    def __init__(self):
        self.output_dir = os.path.join(folder_paths.get_output_directory(),'encryptd')
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "password":  ("STRING", {"default": "123qwe"}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                },
        "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
        
    RETURN_TYPES = ()
    FUNCTION = 'set_password'
    
    OUTPUT_NODE = True

    CATEGORY = "utils"
    
    def set_password(self,images,password,filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        global _password
        _password = password
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": os.path.join('encryptd',subfolder),
                "type": self.type,
                'channel':'rgb'
            })
            counter += 1

        return { "ui": { "images": results} }
    
import torch

class DecryptImage:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "password": ("STRING", {"default": "123qwe"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = 'decrypt_image'
    
    CATEGORY = "utils"
    
    def decrypt_image(self, image, password):
        # 获取图像数据并处理可能的维度问题
        img_data = image.cpu().numpy()
        
        # 处理不同形状的输入
        if len(img_data.shape) == 4 and img_data.shape[0] == 1:
            # 处理形状为 (1, 1, height, width) 的情况
            if img_data.shape[1] == 1:
                img_data = img_data[0, 0]
            else:
                img_data = img_data[0]
        elif len(img_data.shape) == 3 and img_data.shape[0] == 1:
            # 处理形状为 (1, height, width) 的情况
            img_data = img_data[0]
        
        # 处理数据类型问题 - 确保是浮点数并在 0-1 范围内
        if np.issubdtype(img_data.dtype, np.integer):
            img_data = img_data.astype(np.float32) / 255.0
        
        # 转换为 PIL 图像格式
        i = 255. * img_data
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # 检查图像是否已经加密
        # 注意：从 ComfyUI 节点传入的图像可能没有原始的 PNG 元数据
        # 因此我们需要尝试解密
        try:
            # 尝试使用 v2 解密（当前版本）
            dencrypt_image_v2(img, get_sha256(password))
        except Exception as e:
            # 如果失败，尝试使用 v1 解密
            try:
                dencrypt_image(img, get_sha256(password))
            except Exception:
                # 如果都失败，返回原始图像
                pass
        
        # 将解密后的 PIL 图像转换回 ComfyUI 的图像格式 (RGB 通道)
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # 确保图像是 RGB 格式
        if len(img_array.shape) == 2:
            # 灰度图像转换为 RGB
            img_array = np.stack([img_array, img_array, img_array], axis=-1)
        elif img_array.shape[2] == 4:
            # RGBA 转换为 RGB
            img_array = img_array[:, :, :3]
        
        # 确保输出格式符合 ComfyUI 的要求 (1, height, width, 3)
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        # 转换为 PyTorch 张量以确保兼容性
        return (torch.from_numpy(img_array), )

NODE_CLASS_MAPPINGS = {
    "EncryptImage": EncryptImage,
    "DecryptImage": DecryptImage
}
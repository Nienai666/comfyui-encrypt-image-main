
import os
import torch
import json
from PIL import Image as PILImage
from PIL import PngImagePlugin, _util, ImagePalette
from pathlib import Path
from urllib.parse import unquote
import numpy as np
import sys
from typing import Optional

# 导入ComfyUI相关模块
import folder_paths
from comfy.cli_args import args

# 导入核心加密解密功能
from .core.core import get_sha256, dencrypt_image, dencrypt_image_v2, encrypt_image_v2

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
            img = PILImage.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngImagePlugin.PngInfo()
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
    
# 解密图片节点类，用于从本地导入加密图片并解密
class DecryptImageFromFile:
    @classmethod
    def INPUT_TYPES(s):
        # 使用COMBO类型来提供文件选择功能，类似于原生的文件上传节点
        # 获取输入文件夹路径
        input_dir = folder_paths.get_input_directory()
        # 获取所有图片文件
        files = []
        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                if os.path.isfile(os.path.join(input_dir, f)):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in ('.png', '.jpg', '.jpeg', '.webp'):
                        files.append(f)
        
        return {
            "required": {
                "image": (files, {"image_upload": True, "default": files[0] if files else ""}),
                "password": ("STRING", {"default": "123qwe", "placeholder": "输入解密密码"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decrypt_image"
    CATEGORY = "utils"
    
    def decrypt_image(self, image, password):
        # 检查文件是否有效
        if not image:
            raise FileNotFoundError(f"请选择一个有效的图片文件")
        
        try:
            # 获取完整的文件路径
            input_dir = folder_paths.get_input_directory()
            image_path = os.path.join(input_dir, image)
            
            # 使用局部密码解密，不修改全局密码
            # 直接加载图片
            with PILImage.open(image_path) as img:
                img_copy = img.copy()
                
                # 检查并解密图片
                if 'Encrypt' in img_copy.info:
                    if img_copy.info['Encrypt'] == 'pixel_shuffle':
                        dencrypt_image(img_copy, get_sha256(password))
                    elif img_copy.info['Encrypt'] == 'pixel_shuffle_2':
                        dencrypt_image_v2(img_copy, get_sha256(password))
                
                # 转换为数组并处理格式
                img_array = np.array(img_copy)
                
                # 确保是RGB格式
                if len(img_array.shape) == 2:
                    # 灰度图转RGB
                    img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[2] == 4:
                    # RGBA转RGB
                    img_array = img_array[:, :, :3]
                
                # 确保形状符合ComfyUI要求 (1, height, width, 3)
                if len(img_array.shape) == 3:
                    img_array = np.expand_dims(img_array, axis=0)
                
                # 转换为float32并归一化到[0,1]
                img_array = img_array.astype(np.float32) / 255.0
                
                # 转换为PyTorch张量
                return (torch.from_numpy(img_array),)
        except Exception as e:
            print(f"解密图片时出错: {str(e)}")
            # 返回一个空白图像作为错误处理
            blank_image = np.zeros((1, 64, 64, 3), dtype=np.float32)
            return (torch.from_numpy(blank_image),)

# 添加中文节点名称
class 解密图片(DecryptImageFromFile):
    CATEGORY = "工具"

NODE_CLASS_MAPPINGS = {
    "EncryptImage": EncryptImage,
    "DecryptImageFromFile": DecryptImageFromFile,
    "解密图片": 解密图片
}
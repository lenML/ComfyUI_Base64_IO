import torch
import io
import base64
import logging

# 通用工具函数
def serialize_to_base64(data):
    """将 PyTorch 对象序列化为 Base64 字符串"""
    try:
        buffer = io.BytesIO()
        # map_location='cpu' 确保保存时不带设备信息，虽然 save 主要存数据，但保持干净较好
        torch.save(data, buffer)
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return b64_str
    except Exception as e:
        logging.error(f"Serialization failed: {e}")
        return ""

def deserialize_from_base64(b64_str):
    """将 Base64 字符串反序列化为 PyTorch 对象"""
    try:
        if not b64_str:
            return None
        decoded_bytes = base64.b64decode(b64_str)
        buffer = io.BytesIO(decoded_bytes)
        # 强制加载到 CPU，避免显存分配问题
        data = torch.load(buffer, map_location="cpu", weights_only=False)
        return data
    except Exception as e:
        logging.error(f"Deserialization failed: {e}")
        return None

class AnyToBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": ("*",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("b64_string",)
    FUNCTION = "encode"
    CATEGORY = "Base64_IO"
    OUTPUT_NODE = True # 标记为输出节点，确保 API 能获取到结果

    def encode(self, any):
        b64_str = serialize_to_base64(any)
        # 为了让 API 能够方便地获取结果，我们将其作为 UI 输出返回
        return {"ui": {"text": [b64_str]}, "result": (b64_str,)}

class Base64ToLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "b64_string": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "decode"
    CATEGORY = "Base64_IO"

    def decode(self, b64_string):
        latent = deserialize_from_base64(b64_string)
        if latent is None:
            # 返回一个空的 Latent 结构防止报错，或者直接抛出异常
            # 这里生成一个极小的 1x1 Latent 作为 fallback
            logging.warning("Base64ToLatent: Input is empty or invalid, returning empty latent.")
            return ({"samples": torch.zeros((1, 4, 8, 8))}, )
        return (latent,)

class Base64ToConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "b64_string": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "decode"
    CATEGORY = "Base64_IO"

    def decode(self, b64_string):
        cond = deserialize_from_base64(b64_string)
        if cond is None:
            logging.error("Base64ToConditioning: Failed to decode data.")
            raise ValueError("Invalid Base64 string for Conditioning")
        return (cond,)

class Base64ToAny:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "b64_string": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            }
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "decode"
    CATEGORY = "Base64_IO"

    def decode(self, b64_string):
        data = deserialize_from_base64(b64_string)
        if data is None:
            logging.error("Base64ToAny: Failed to decode data.")
            raise ValueError("Invalid Base64 string")
        return (data,)

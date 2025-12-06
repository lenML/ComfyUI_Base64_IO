from .nodes import *



NODE_CLASS_MAPPINGS = {
    "AnyToBase64": AnyToBase64,
    "Base64ToAny": Base64ToAny,
    "Base64ToLatent": Base64ToLatent,
    "Base64ToConditioning": Base64ToConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyToBase64": "Any -> Base64",
    "Base64ToAny": "Base64 -> Any",
    "Base64ToLatent": "Base64 -> Latent",
    "Base64ToConditioning": "Base64 -> Conditioning",
}


__all__ = ['NODE_CLASS_MAPPINGS']

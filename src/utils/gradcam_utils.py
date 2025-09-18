import numpy as np, cv2, torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def vit_reshape_transform(tensor, h=14, w=14):
    result = tensor[:, 1:, :].transpose(1,2).reshape(tensor.size(0), -1, h, w)
    return result

def get_cam(model, target_layers, is_vit=False):
    return GradCAM(model=model, target_layers=target_layers,
                   reshape_transform=vit_reshape_transform if is_vit else None)

def overlay_cam(pil_img, grayscale_cam):
    import numpy as np
    img = np.float32(pil_img)/255.0
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return visualization

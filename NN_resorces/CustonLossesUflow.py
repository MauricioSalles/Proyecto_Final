import torch
import torchvision

def edge_weighting_fn(x):
    return torch.exp(-torch.mean(
        input=((150 * x)**2),
        dim=1, keepdim=True))

def second_order_smoothness_loss(image, flow,edge_weighting_fn):
  img_gx, img_gy = image_grads(image, stride=2)
  weights_xx = edge_weighting_fn(img_gx)
  weights_yy = edge_weighting_fn(img_gy)
  flow_gx, flow_gy = image_grads(flow)
  flow_gxx, _ = image_grads(flow_gx)
  _, flow_gyy = image_grads(flow_gy)
  return ((torch.mean(input=weights_xx * robust_l1(flow_gxx)) +
           torch.mean(input=weights_yy * robust_l1(flow_gyy))) / 2.)


def image_grads(image_batch, stride=1):
  image_batch_gh = image_batch[:,:, stride:] - image_batch[:,:, :-stride]
  image_batch_gw = image_batch[:,:, :, stride:] - image_batch[:,:, :, :-stride]
  return torch.nn.functional.pad(input=image_batch_gw,pad=(0, stride,0, 0)),torch.nn.functional.pad(input=image_batch_gh,pad=(0, 0,stride, 0))

def robust_l1(x):
  return (x**2 + 0.001**2)**0.5

def census_transform(image, patch_size):
  intensities = torchvision.transforms.functional.rgb_to_grayscale(image,1) * 255
  kernel = torch.reshape(
      torch.eye(patch_size * patch_size),
      (patch_size * patch_size, 1, patch_size, patch_size)).to(image.device)
  neighbors = torch.nn.functional.conv2d(
      input=intensities, weight=kernel, padding=3)
  diff = neighbors - intensities
  diff_norm = diff / torch.sqrt(.81 + torch.square(diff))
  return diff_norm

def soft_hamming(a_bhwk, b_bhwk, thresh=.1):
  sq_dist_bhwk = torch.square(a_bhwk - b_bhwk)
  soft_thresh_dist_bhwk = sq_dist_bhwk / (thresh + sq_dist_bhwk)
  return torch.sum(
      input=soft_thresh_dist_bhwk, dim=1,keepdim=True)
  
def census_loss(image_a_bhw3,
                image_b_bhw3,
                mask_bhw3,
                patch_size=7):
  eps=0.01
  q=0.4
  census_image_a_bhwk = census_transform(image_a_bhw3, patch_size)
  census_image_b_bhwk = census_transform(image_b_bhw3, patch_size)
  hamming_bhw1 = soft_hamming(census_image_a_bhwk,
                              census_image_b_bhwk)
  padded_mask_bhw3 = zero_mask_border(mask_bhw3, patch_size)
  diff = torch.pow((torch.abs(hamming_bhw1) + eps), q)
  diff *= padded_mask_bhw3
  diff_sum = torch.sum(input=diff)
  loss_mean = diff_sum / (torch.mean(torch.sum(
      input=padded_mask_bhw3.detach() + 1e-6)))
  return loss_mean

def zero_mask_border(mask_bhw3, patch_size):
  mask_padding = patch_size // 2
  mask = mask_bhw3[:,:, mask_padding:-mask_padding, mask_padding:-mask_padding]
  return torch.nn.functional.pad(
      input=mask,
      pad=(mask_padding, mask_padding,mask_padding, mask_padding))
  
def flow_to_warp(flow):
  N, C, H, W = flow.shape
  coords = torch.meshgrid(torch.arange(H, device=flow.device), torch.arange(W, device=flow.device))
  coords = torch.stack(coords[::-1], dim=0).float()
  coords = coords[None].repeat(C, 1, 1, 1)
  warp = coords + flow
  return warp

def resample(source, coords):
  N, C, H, W = source.shape
  coords = coords.permute(0, 2, 3, 1)
  xgrid, ygrid = coords.split([1,1], dim=3)
  xgrid = 2*xgrid/(W-1) - 1
  ygrid = 2*ygrid/(H-1) - 1
  grid = torch.cat([xgrid, ygrid], dim=-1)
  output = torch.nn.functional.grid_sample(source, grid, align_corners=True)
  return output

def mask_invalid(coords, pad_h=0, pad_w=0):
  pad_h = float(pad_h)
  pad_w = float(pad_w)
  max_height = float(coords.shape[3] - 1)
  max_width = float(coords.shape[2] - 1)
  mask = torch.logical_and(
      torch.logical_and(coords[:, 0, :, :] >= pad_h,
                     coords[:, 0, :, :] <= max_height),
      torch.logical_and(coords[:, 1, :, :] >= pad_w,
                     coords[:, 1, :, :] <= max_width))
  mask = mask.type(torch.float32)[:, None,:, :]
  return mask

def self_supervision_loss(teacher_flow, 
                          student_flow,
                          teacher_backward_flow,
                          student_backward_flow):

  h = teacher_flow.shape[2]
  w = teacher_flow.shape[3]

  student_warp = flow_to_warp(student_flow)
  student_backward_flow_resampled = resample(student_backward_flow,student_warp)
  teacher_warp = flow_to_warp(teacher_flow)
  teacher_backward_flow_resampled = resample(teacher_backward_flow,teacher_warp)
  student_valid_warp_masks = mask_invalid(student_warp)
  teacher_valid_warp_masks = mask_invalid(teacher_warp)

  student_fb_sq_diff = torch.mean(torch.sum(
      (student_flow + student_backward_flow_resampled)**2,
      axis=-1,
      keepdims=True))
  teacher_fb_sq_diff = torch.mean(torch.sum(
      (teacher_flow + teacher_backward_flow_resampled)**2,
      axis=-1,
      keepdims=True))
  student_fb_consistency = torch.exp(-student_fb_sq_diff / (0.03**2 *
                                                          (h**2 + w**2)))
  teacher_fb_consistency = torch.exp(-teacher_fb_sq_diff / (0.003**2 *
                                                          (h**2 + w**2)))

  student_mask = 1. - (student_fb_consistency * student_valid_warp_masks)
  teacher_mask = teacher_fb_consistency * teacher_valid_warp_masks
  #teacher_mask = selfsup_transform_fn(teacher_mask, is_flow=False)
  #teacher_flow = selfsup_transform_fn(teacher_flow, is_flow=True)

  error = robust_l1(teacher_flow.detach() - student_flow)
  mask = (teacher_mask * student_mask).detach()

  return torch.mean(input=mask * error)

def compute_occlusions(forward_flow):
  B,C,H,W = forward_flow.shape
  flow_ij = forward_flow
  occlusion_mask = torch.ones((B,1,H,W), dtype=torch.float32, device=forward_flow.device)
  warp = flow_to_warp(occlusion_mask)
  occlusion_mask = resample(occlusion_mask,warp)  
  occlusion_mask = torch.minimum(occlusion_mask, mask_invalid(warp))
  return occlusion_mask


def unsupLoss(Frame1,Frame3,Warp,Flow12,Flow21,Flow12T,Flow21T):
  mask_level0 = compute_occlusions(forward_flow=Flow12)
  losses = 0
  cen_loss = census_loss(
      image_a_bhw3=Frame3,
      image_b_bhw3=Warp,
      mask_bhw3=mask_level0)
  losses += cen_loss
  smooth_loss_2nd = second_order_smoothness_loss(
      image=Frame1,
      flow=Flow12,
      edge_weighting_fn=edge_weighting_fn)
  losses += smooth_loss_2nd * 4
  
  selfsup_loss = self_supervision_loss(
      teacher_flow=Flow12,
      student_flow=Flow12T,
      teacher_backward_flow=Flow21,
      student_backward_flow=Flow21T)
  losses += selfsup_loss * 0.3
  
  return losses
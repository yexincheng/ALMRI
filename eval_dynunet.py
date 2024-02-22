import argparse
from glob import glob
import os
from monai.transforms import MapTransform, Resize
from typing import Dict, Hashable, Mapping, Optional, Sequence, Union
from monai.utils import InterpolateMode, ensure_tuple_rep
from monai.config import KeysCollection
from monai.data import MetaTensor
import distutils.util
import nibabel as nib
import torch
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
    SqueezeDimd,
    ToNumpyd,
    Compose,
    AddChanneld,
    CenterSpatialCropd,
    Invertd,
)
# from monailabel.transform.post import Restored
from monai.apps.deepedit.transforms import (
    AddGuidanceFromPointsDeepEditd,
    AddGuidanceSignalDeepEditd,
    DiscardAddGuidanced,
    ResizeGuidanceMultipleLabelDeepEditd,
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import DynUNet
from monai.handlers.utils import from_engine
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import SimpleITK as sitk
from metrics import compute_dice_coefficient
import warnings
warnings.filterwarnings("ignore")


class Restored(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        ref_image: str,
        has_channel: bool = True,
        mode: str = InterpolateMode.NEAREST,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        meta_key_postfix: str = "meta_dict",
    ):
        super().__init__(keys)
        self.ref_image = ref_image
        self.has_channel = has_channel
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.meta_key_postfix = meta_key_postfix

    def __call__(self, data):
        d = dict(data)
        meta_dict = (
            d[self.ref_image].meta
            if d.get(self.ref_image) is not None and isinstance(d[self.ref_image], MetaTensor)
            else d.get(f"{self.ref_image}_{self.meta_key_postfix}", {})
        )

        for idx, key in enumerate(self.keys):
            result = d[key]
            current_size = result.shape[1:] if self.has_channel else result.shape
            spatial_shape = meta_dict.get("spatial_shape", current_size)
            spatial_size = spatial_shape[-len(current_size) :]
            # Undo Spacing
            if torch.any(np.not_equal(current_size, spatial_size)):
                
                resizer = Resize(spatial_size=spatial_size, mode=self.mode[idx])
                result = resizer(result, mode=self.mode[idx], align_corners=self.align_corners[idx])
            d[key] = result if len(result.shape) <= 3 else result[0] if result.shape[0] == 1 else result

            meta = d.get(f"{key}_{self.meta_key_postfix}")
            if meta is None:
                meta = dict()
                d[f"{key}_{self.meta_key_postfix}"] = meta
            meta["affine"] = meta_dict.get("original_affine")
        return d

def get_pre_transforms(labels, spatial_size, number_intensity_ch):

	pre_transforms = Compose(
		[
		    LoadImaged(keys="image"),
		    EnsureChannelFirstd(keys="image"),
		    Orientationd(keys="image", axcodes="RAS"),
		    AddGuidanceFromPointsDeepEditd(ref_image="image", guidance="guidance", label_names=labels),
		    CenterSpatialCropd(keys="image", roi_size=spatial_size),
            #Resized(keys="image", spatial_size=spatial_size, mode="area"),
		    ResizeGuidanceMultipleLabelDeepEditd(guidance="guidance", ref_image="image"),
		    AddGuidanceSignalDeepEditd(keys="image", guidance="guidance", number_intensity_ch=number_intensity_ch),
		    EnsureTyped(keys="image")
		]
	)
	return pre_transforms

def get_post_transforms(pre_transforms):
	post_transforms = Compose([
        EnsureTyped(keys="pred"),
        #Orientationd(keys=('image',"pred"), axcodes="LAS"),
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True),
        Invertd(keys="pred", transform=pre_transforms, orig_keys="image"),
        #SqueezeDimd(keys="pred", dim=0),
        #ToNumpyd(keys="pred"),
        #AddChanneld(keys='pred'),
        Restored(keys="pred", ref_image="image", mode="nearest"),
    ])
	return post_transforms

def get_network(labels, number_intensity_ch):
	model = DynUNet(
		spatial_dims=3,
		in_channels=len(labels) + number_intensity_ch, 
		out_channels=len(labels), 
		kernel_size=[3, 3, 3, 3, 3, 3],
		strides=[1, 2, 2, 2, 2, [2, 2, 1]],
		upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
		norm_name="instance",
		deep_supervision=False,
		res_block=True,
	)
	return model

def data_load(args, pre_transforms):
	testing_pool_path = os.path.join(args.prefix, args.task, args.mode)
	testing_pool = glob(os.path.join(testing_pool_path, '*.npz'))
	testing_pool_nii = [(os.path.basename(sample).split('.npz')[0]+'.nii.gz') for sample in testing_pool]
	imagelist = sorted([os.path.join(args.prefix, 'trainingset_51', image_name) for image_name in testing_pool_nii])

	# images = sorted(glob(os.path.join(args.dataDir, '*.nii.gz')))

	imagesd = [{'image':i} for i in imagelist]
	images_ds = Dataset(data=imagesd, transform=pre_transforms)
	images_loader = DataLoader(images_ds, batch_size=1,shuffle=False)

	return images_loader, imagelist

def infer(args):

	save_path_ckp = os.path.join(args.root, 'checkpoints', args.base_model + '_' + args.task + '_' + args.strategy, f'seed{args.seed}', f'epochs{args.num_epochs}')
	ckp_pool = sorted(glob(os.path.join(save_path_ckp, '*latest.pth')))
	print(save_path_ckp)
	print('Number of checkpoints: ', len(ckp_pool))
	save_nii_path = os.path.join('datasets/predictions', f'{args.task}_{args.base_model}_{args.strategy}', f'seed{args.seed}', f'epochs{args.num_epochs}', args.mode)
	os.makedirs(save_nii_path, exist_ok=True)
	
	save_path_dice = os.path.join('results/dice', f'epochs{args.num_epochs}')
	os.makedirs(save_path_dice, exist_ok=True)

	device = torch.device("cuda" if args.use_gpu else "cpu")
	pre_transforms = get_pre_transforms(args.labels, args.spatial_size, args.number_intensity_ch)
	post_transforms = get_post_transforms(pre_transforms)

	dataloader, imagelist = data_load(args, pre_transforms)
	#print(len(dataloader))
	dice_test = []
	network = get_network(args.labels, args.number_intensity_ch)
	for j, p in enumerate(tqdm(ckp_pool)):
		print('running checkpoint', p)
		avg_dice = []
		network.load_state_dict(torch.load(p))
		network.to(device)
		network.eval()
		with torch.no_grad():
			print('Strat infering ')
			for ii,i in enumerate(dataloader):
				original_affine = i["image_meta_dict"]["affine"][0,:,:].numpy()
				# print(original_affine.shape)
				name = os.path.split(imagelist[ii])[1]
				print(name)
				gt_path = os.path.join(args.prefix, 'trainingset_51', 'labels/final', name)
				gt = sitk.ReadImage(gt_path)
				gt_array = sitk.GetArrayFromImage(gt)
				gt_array = np.transpose(np.uint8(gt_array==args.label_id),(2,1,0))
				
				# input = i["image"].repeat(1, 2, 1, 1, 1).to(device)
				input = i["image"].to(device)
			
				i['pred'] = network(input)[0]
				i['image'] = i['image'][0]  
				#i = pre_transforms.inverse(i)
				i = post_transforms(i)
				# print('shapes', i['pred'].astype(np.uint8).shape, gt_array.shape)
				# print('types', i['pred'].astype(np.uint8).dtype, gt_array.dtype, 'unique', np.unique(i['pred'].astype(np.uint8)), np.unique(gt_array))
				dice = compute_dice_coefficient(gt_array, i['pred'].astype(np.uint8))
				avg_dice.append(dice)
				if '40' in p:
					nib.save(
							nib.Nifti1Image(i['pred'].astype(np.uint8), original_affine), os.path.join(save_nii_path, name)
						)
			np.save(os.path.join(save_path_dice, 
					  f'{args.strategy}_{args.task}_{args.base_model}_epochs{args.num_epochs}_seed{args.seed}_{args.mode}_{j+1}samples_dice.npy'), avg_dice)
			dice_test.append(np.mean(avg_dice))
				
	# save dice
	np.save(os.path.join(save_path_dice, 
					  f'{args.strategy}_{args.task}_{args.base_model}_epochs{args.num_epochs}_seed{args.seed}_{args.mode}_dice.npy'), dice_test)
    # plot dice curve
	plt.plot(dice_test)
	plt.xlabel('Number of samples')
	plt.ylabel('Dice coefficient')
	plt.title(f'{args.base_model} {args.task} {args.strategy} epochs{args.num_epochs} seed{args.seed} sampling')
	plt.savefig(os.path.join(f'figures/{args.mode}', f'{args.base_model}_{args.task}_{args.strategy}_epochs{args.num_epochs}_seed{args.seed}_sampling.png'))
	plt.close()

def strtobool(val):
    return bool(distutils.util.strtobool(val))


def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-g", "--use_gpu", type=strtobool, default="true")
	parser.add_argument("--size", nargs=3, type=int)
	parser.add_argument('--root', type=str, default='./', help='root path for checkpoints')
	parser.add_argument('--prefix', type=str, default='./datasets/RAINE_organ_51', help='dataset path prefix')
	parser.add_argument('--task', type=str, default='MRI_Pancreas', help='task name')
	parser.add_argument('--strategy', type=str, default='random', help='sampling strategy')
	parser.add_argument('--base_model', type=str, default='DynUNet', help='base model name')
	parser.add_argument('--label_id', type=int, default=3, help='label id')
	parser.add_argument('--seed', type=int, default=2023, help='random seed')
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--mode', type=str, default='test', help='mode: test or train')

	args = parser.parse_args()
	args.labels = {
		"left kidney": 1,
		# "right kidney": 2,
		# "pancreas": 3, 
		"background": 0,
	}

	args.spatial_size=(args.size)
	# target_spacing=(1.0, 1.0, 1.0)
	args.number_intensity_ch=1
	
	infer(args)
	print('Finished')      
  

if __name__ == '__main__':
	main()
	


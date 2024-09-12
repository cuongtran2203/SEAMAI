
import torch
from torchvision import transforms
class Compose(object):
	"""Composes several transforms together.

	Args:
		transforms (list of transform objects): list of data transforms to compose.
	"""

	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, img):
		for t in self.transforms:
			img = t(img)
		return img

	def randomize_parameters(self):
		for t in self.transforms:
			if hasattr(t, 'randomize_parameters'):
				t.randomize_parameters()

class ToTensor(object):
	"""Convert a tensor to torch.FloatTensor in the range [0.0, 1.0].

	Args:
		norm_value (int): the max value of the input image tensor, default to 255.
	"""

	def __init__(self, norm_value=255):
		self.norm_value = norm_value

	def __call__(self, pic):
		if isinstance(pic, torch.Tensor):
			return pic.float().div(self.norm_value)

	def randomize_parameters(self):
		pass

def transforms_train(img_size=224, distortion_p=0.5, hflip=0.5, color_jitter=0.01):

	primary_tfl = []
	if hflip > 0.:
		primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]

	if color_jitter is not None:
		primary_tfl += [transforms.ColorJitter(saturation=color_jitter, hue=color_jitter)]

	primary_tfl += [transforms.RandomAutocontrast(p=0.5),
				#transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.01, 5))], p=0.5),
				transforms.RandomGrayscale(p=0.5),
				transforms.RandomPerspective(distortion_scale=0.1, p=distortion_p),
				]
	primary_tfl += [
		transforms.Resize(img_size)
		#transforms.RandomCrop(img_size)
		]

	#final_tfl = []
	#final_tfl += [
	#	ToTensor()
	#]
	return Compose(primary_tfl)


def transforms_eval(img_size=224):

	tfl = [
		transforms.Resize(img_size),
		#transforms.CenterCrop(img_size)
	]
	#tfl += [
	#	ToTensor()
	#]

	return Compose(tfl)


def create_video_transform(input_size=224, is_training=False, distortion=None, hflip=0.5, color_jitter=0.4):

	if isinstance(input_size, (tuple, list)):
		img_size = input_size[-2:]
	else:
		img_size = input_size

	if is_training:
		transform = transforms_train(img_size, distortion_p=distortion, hflip=hflip, color_jitter=color_jitter)
	else:
		transform = transforms_eval(img_size)

	return transform

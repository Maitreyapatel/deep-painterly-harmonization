from PIL import Image
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--style", help="style image path",
                    type=str, required=True)
parser.add_argument("--input", help="input image path",
                    type=str, required=True)
parser.add_argument("--mask", help="tight mask path", type=str, required=True)
parser.add_argument("--loose_mask", help="loose mask path",
                    type=str, required=True)
args = parser.parse_args()


style_img = Image.open(args.style)
input_img = Image.open(args.input)
mask_img = Image.open(args.mask)
l_mask_image = Image.open(args.loose_mask)

fig, a = plt.subplots(1, 4, figsize=(10, 10))

a[0].imshow(style_img)
a[0].axis('off')

a[1].imshow(input_img)
a[1].axis('off')

a[2].imshow(input_img * mask_image)
a[2].axis('off')

a[3].imshow(input_img * l_mask_image)
a[3].axis('off')

plt.show()

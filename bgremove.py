#import pixellib
import pixellib

import os

#from pixellib we imported in the class alter_bg
from pixellib.tune_bg import alter_bg

change_bg = alter_bg(model_type = "pb")

#load the deeplabv3+ model
change_bg.load_pascalvoc_model("D:\\xception_pascalvoc.pb")

#f_image_path:the image which background would be changed. (original image)
#b_image_path: background image for original image
#output_image_name: The new image with a changed background.
change_bg.change_bg_img(f_image_path = "D:\\original.jpg",b_image_path = "D:\\back.jpg", output_image_name="D:\\new_img.jpg")

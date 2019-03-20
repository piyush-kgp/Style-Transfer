

from style_transfer.stylize import StyleTransfer
from style_transfer.algorithms import cyclegan as CYCLEGAN
from style_transfer.algorithms import johnson as JOHNSON
from style_transfer.algorithms import gatys as GATYS
from style_transfer.prepro.prepro import imread
from style_transfer.prepro.prepro import imsave
from style_transfer.stylizer import StyleTransfer
import style_transfer as stl

ADAPTIVE = 'ADAPTIVE'

def stylize(content_img, style_img):
    stylizer = StyleTransfer(model=stl.GATYS)
    stylizer.do_stuff()

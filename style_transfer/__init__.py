
from style_transfer import gatys
from style_transfer import cyclegan
import stylize

class StyleTransfer:
    def __init__(self, stylizer='gatys'):
        if stylizer=='gatys':
            model = gatys
        elif stylizer=='cyclegan':
            model = cyclegan
    

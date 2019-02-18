
class StyleTransfer:
    def __init__(self, stylizer='gatys'):
        if stylizer=='gatys':
            model = gatys
        elif stylizer=='cyclegan':
            model = cyclegan
    def do_stuff(self):
        pass


import style_transfer as stl

def custom_test():
    content_img = stl.imread('big_apple.jpeg')
    style_img = stl.imread('starry_nights.jpeg')
    stylized_img = stl.stylize(content_img, style_img)
    stl.imsave('stylized.jpg', stylized_img)

custom_test()

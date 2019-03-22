
### Intro
Intro goes here

### Installation
Installation is as easy as:
`pip install style_transfer`

If that raises doesn't work, please use this command:
`pip install git+__`

### List of available Style Transfer algorithms
Algorithms implemented can be checked by `stl.algorithms`
At present we support
- [Gatys' Neural Style Transfer](https://arxiv/__.pdf)
- [CycleGAN]()
- [Johnson's Algorithm]()
- [Adaptive StyleGAN]()
If you want to understand how these algorithms work, [this blog]() written during the development of this library may be helpful.

### Usage
We provide some style images as part of the library, so you don't have to download the images for use. Available images can be checked by `stl.available_style_imgs`. They can be used directly to stylize your image:
```
import style_transfer as stl
content_img = stl.imread(fp) # Please replace fp with your file path
stylized_img = stl.stylize(content_img, stl.STARRY_NIGHTS)
stl.imsave(fp, stylized_img)
```

To quickly check the algorithm performance, we also provide some content images. This list can be checked by `stl.available_content_imgs`.
So, all you need to do is 1 line:
`stl.stylize(stl.BIG_APPLE, stl.STARRY_NIGHTS).imsave('big_apple_starry_nights.jpg')`

##### Custom content and style image
```
import style_transfer as stl
content_img = stl.imread(fp) # Please replace fp with your file path
style_img = stl.imread(fp) #Please replace fp with your file path
stylized_img = stl.stylize(content_img, style_img)
stl.imsave(fp, stylized_img)
```

### Help
We have a documentation at __

If that doesn't resolve the problem, please raise an issue while following __

### Development
1. Clone this repo.
2. Create and run a virtual environment.
4. cd into the library and modify source code as you see fit.
3. Install the library in the dev mode
5. Submit a PR if you find a bug/have an improvement. If it is your first time with Open Source Contribution, [this]() may help.

```
git clone __
virtualenv venv && source venv/bin/activate
cd StyleTransfer
python setup.py install
```

### Docker
We also provide a [docker image]() for quick installation and run.
<!-- Create a docker image with all dependencies installed and all pb files and weights.
This can be used to quickly test the library. -->

### Citations
```
Citations go here
```

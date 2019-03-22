
### Gatys' Algorithm
1. Start with a content image C, a style image S and a random image G
2. Use a pretrained CNN model (authors used VGG) graph  in Tensorflow and mark first 4 conv nodes
3. Define your content cost as the L2 norm between conv4_2 node outputs of C and G
4. Pass S and G to the graph and define style cost as the weighted sum of L2 norms of
gram matrices at those 4 conv nodes between S and G
4. Define the cost function as weighted sum of the content and style costs and the optimizer using Adam.
5. Update the generated image by minimizing the cost.
6. Repeat step 5 for some finite number (1000) of times after which hopefully we have a stylized image.

# Part 1: Parse the DICOM images and Contour Files

This is implemented as `DataLoader.__load_sample()`.

*How did you verify that you are parsing the contours correctly?*

See `test_parsing.py`, which overlays masks over original images and 4 example results (see 4 images in test_data). Also, I use logging for errors and the dataset could be loaded without any errors.

*What changes did you make to the code, if any, in order to integrate it into our production code base?*

None, the research team did a good job;).


# Part 2: Model training pipeline

This is implemented as `DataLoader` class.

*Did you change anything from the pipelines built in Parts 1 to better streamline the pipeline built in Part 2? If so, what? If not, is there anything that you can imagine changing in the future?*

I started working only when I read the goal of both Part 1 and 2, so no changes were necessary. In the future, I guess one will also read the outer contours as another set of targets. The changes will depend on the specified behavior in what to do with images which have either inner or outer contour but not both.

*How do you/did you verify that the pipeline was working correctly?*

See `test_dataloader.py`. I created a small test dataset and asserted the expected behavior, as specified by you, there. 

*Given the pipeline you have built, can you see any deficiencies that you would change if you had more time? If not, can you think of any improvements/enhancements to the pipeline that you could build in?*

One can perhaps add __iter__ method or preload all images into memory.

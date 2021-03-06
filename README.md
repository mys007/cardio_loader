# Part 1: Parse the DICOM images and Contour Files

This is implemented as `DataLoader.__load_sample()`.

*How did you verify that you are parsing the contours correctly?*

See `test_parsing.py`, which overlays masks over original images and 4 example results (see 4 png images in test_data).

*What changes did you make to the code, if any, in order to integrate it into our production code base?*

None, the research team did a good job (docs, exception handling,...) ;).


# Part 2: Model training pipeline

This is implemented as `DataLoader` class. I interpreted your third requirement as reshuffling instead of random sampling.

*Did you change anything from the pipelines built in Parts 1 to better streamline the pipeline built in Part 2? If so, what? If not, is there anything that you can imagine changing in the future?*

I started working only when I read the goal of both Part 1 and 2, so no changes were necessary. In the future, I guess one will also read the outer contours as another set of targets. The changes will depend on the specified behavior in what to do with images which have either inner or outer contour but not both.

*How do you/did you verify that the pipeline was working correctly?*

See `test_dataloader.py`. I created a small test dataset and asserted the expected behavior there. 

*Given the pipeline you have built, can you see any deficiencies that you would change if you had more time? If not, can you think of any improvements/enhancements to the pipeline that you could build in?*

One can perhaps add `__iter__` method or preload all images into memory. Of course, many unit test may be added (including mocks), but I guess this was not the point of the exercise.


# Part 3: Parse the o-contours

The new requirement caused some changes. The new class can now handle both regimes (returning images with inner contour or images with both contours). I also had to change the public interface `next()` so that it returns triplets instead of pairs. In production if no change to interface was allowed, one could add a different method or derive a class. I also updated unit tests and added a testcase.

# Part 4: Heuristic LV Segmentation approaches

Please see `segmentation.ipynb`.
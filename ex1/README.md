Update 05/11 - 10am - Kathi:
So far I have:
- created generating Dataset function
- plotting original data 
- implemented all models (without MMD so far)

My next goals are to:
- implement MMD and plot it against each each other
- unify plotting to make it look pretty!
- add a section for comments at the end of the notebook
- do the visualization for the hyperparameters and choose good and bad examples


Notes and Thoughts:
- Bin sizes: what kinda sizes actually make sense? do we need to consider special cases depending on the sample size and maybe throw exceptions?
- adjusted         x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50)) in KDE because I dont have the time for 100 

Update 01/11 - 4pm - Kathi:

- I added a folder for MMD including the main MMD file, a unit test (ran successfully) and an init file so we can call it in our other files
- I added a KDE file that runs the KDE, does two plots. we still need MMD here as well as a double check.
- I updated my Single Gaussian to actually generate samples using inverse transform. I tried to apply MMD here, however, it seems super low compared to my first intuition of it being really bad lol

If someone could:
- implement MMD in other models
- create comments about (dis-)advantages for all models
- double check the MMD and KDE functionality

Next general steps imo:
- Finish Ex 1!!
  - especially the layout.
  - integrate all functions into one notebook
  - Plotting
  - we need different samples (i.e. n=20, 50, 100 ...) for all our models
  - create folders that structure our repo 


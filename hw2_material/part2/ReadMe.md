### Step of Running my code!

1. Run  ```python3 main.py``` to get the model.pt (train() is in main.py)

2. Run ```python3 eval.py``` to get the accuracy of the model.

3. I'm not sure if the code has some "path problem" due to the original set in my local computer. Especially my finder contains the p2_data, and the file myDatasets.py row 107:

   ```read_img = Image.open(self.prefix + '/' + self.images[idx])```

   I have some issues in my local computer at first.

BTW, I used to saving my figures of model training set and validation set.

If it is annoying, you can close it in file tool.py row 60.


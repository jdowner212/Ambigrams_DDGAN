### About pre-trained models

**'RS' models:**

Some RS models had a little bit of luck producing results in earlier epochs, but most did best with 20+ epochs. 
Did not attempt training over 30 epochs, but no there was no indication that it would be a bad idea.

**'AB' model:**

Did best epochs 5+, no evidence past 20 epochs.

**'ZE' models:**

Most starting showing some reasonable results around 6 or 7 epochs and continued to do so, but no evidence past 15 epochs<br><br><br>
For each letter pair (RS, AB, ZE) the differences between models were minimal. However, differences 
between models of different letter pairings varied considerably. This may have affected the above
trends.

The hyperparameters used for each model can be found in `~models/{letters}/{model_name}/log.txt`.


### Using `add_augmented_data()` in Ambigrams.ipynb

I ran this notebook in Colab, and for some reason, if a runtime ended, it wasn't able to properly read in existing data -- 
maybe you'll have better luck. But for this reason I called `add_augmented_data()` with `delete_existing = True` every 
time I started a new session. This took about 5 minutes for each pair of letters.

# %% [markdown]
# # Python tutorial for using Attentive Chrome

# %% [markdown]
# ## 1. Install kipoi
# This can be done by using `pip`. (Ex `pip install kipoi`). Note that you need anaconda or miniconda installed. Refer to https://kipoi.org/docs/#installation for more information.

# %% [markdown]
# ## 2. Using Attentive Chrome in your Python program

# %% [markdown]
# First, we need to import kipoi. Also, we download our example file for the tutorial.

# %%
import kipoi

# %% [markdown]
# Now, suppose we want to predict for cell type E005. We first go ahead and create a model for the cell type.

# %%
model = kipoi.get_model("AttentiveChrome/E005")

# %% [markdown]
# There are two ways to predict using this model object. First is to predict using the pipeline. This makes prediction for all batches.

# %%
dl_dictionary = model.default_dataloader.example_kwargs #This is an example dataloader.
print("dl_dictionary:", dl_dictionary)

prediction = model.pipeline.predict(dl_dictionary)

#If you wish to make prediction on your own dataset, run this code:
#prediction = model.pipeline.predict({"input_file": "path to input file", "bin_size": {some integer}})


# %% [markdown]
# Our output of prediction is a numpy array containing the output from the final softmax function.

# %%
print("Prediction result:")
print(prediction)

# %% [markdown]
# **The following example doesn't work! The code is just kept as an example.**
# 
# Another way to make a prediction is to predict for single batches. We first need to create our dataloader.
# Then, we can create an iterator of fixed batch size. 

# %%
# dl = model.default_dataloader.init_example()
# it = dl.batch_iter(batch_size=32) #iterator of batch size 32

# single_batch = next(it) #this gets us a single batch

# %% [markdown]
# For sake of example, let's make a prediction on the first 10 batches.

# %%
# The batch var isn't defined so the code fails!
# for i in range(10):
#     print("Making prediction on batch",i)
#     prediction = model.predict_on_batch(batch['inputs'])



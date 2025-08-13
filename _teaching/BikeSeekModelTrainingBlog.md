---
title: "BikeSeek Blog - Training the Embedding Model"
collection: teaching
permalink: /ML/BikeSeekModelTraining
excerpt: 'A Summary of the experiments done to train the embedding model used in the search engine of [bikeseek.org](bikeseek.org)'
date: 2025-08-13
---


# The Goal 
According to [Statista](https://de.statista.com/statistik/daten/studie/157410/umfrage/polizeilich-erfasste-faelle-des-diebstahls-von-fahrraedern-seit-1995/), around 250,000 bicycles get reported to the police as stolen every year in Germany alone. Meanwhile as stated in the [press release](https://news.mit.edu/2023/where-do-stolen-bikes-go-0215) of a study conducted by MIT, the Amsterdam Institute for Advanced Metropolitan Solutions and the Delft University of Technology, "A substantial amount \[of stolen bicycles\] appear to get resold". 

Our goal is to create an image-based search engine that scans listings on big ecommerce platforms to potentially find stolen bicycles there. A user that recently got their bicycle stolen uploads an image of their bike to the website and the search engine retrieves visually similar bicycles listed for sale on said platforms; together with some metadata, e.g. location and URL to the listing.  

In this blog we will go into the aspect of training an embedding model to perform this "visual similarity search". We will detail the whole pipeline: From gathering the dataset, training the embedding model, enhancing the search results by training a reranking model to finally deploying the model on a moderate CPU.

# The Basic Idea
## How Image Search Works
One typical way to do image search is to have some model that extracts "features" of a given image, such that when two images that are supposed to be deemed "similar to each other" are considered, their computed features will be very similar. At the same time, when we take to dissimilar images, their features are supposed to be very different from each other. 
Once one has computed these features, so-called embeddings, for all of the images in a database can then use these embeddings to search for the most similar images using [Nearest-Neighbour-Search](https://en.wikipedia.org/wiki/Nearest_neighbor_search#:~:text=Nearest%20neighbor%20search%20(NNS)%2C,the%20larger%20the%20function%20values.). 
![](/images/Drawing 2025-08-13 11.15.11.excalidraw.png)
The main question is then how one would desing a model that learns these meaningful "embeddings". 
In this Blog, I'll be using a Vision Transformer to learn these embeddings. To train the model, we will need to pass it multiple images and reward the model for clustering together similar images and punishing the model for clustering together dissimilar images. But to do this we first need an organized dataset where it is clearly decided which images are supposed to be deemed "similar":
## Gathering the Dataset
To get a good dataset, I did the following: I collected a list of 50 big bike brands and for each of them noted around 10 or so of their available bike models. I then wrote a script that searched the Web for "{Brand Name} {Model Name}" and downloaded the images. What then followed was me manually sorting the results into groups of bicycles with the same colour of the bike frame, e.g. one would have one group for "{Brand Name} {Model Name} Blue" and one group for "{Brand Name} {Model Name} Red". The image within one such group were then deemed "similar" and the goal of the model was to cluster these together while seperating them from the remaining "dissimilar" images.
Here is an example of what one such group looks like: 
![](/images/Screenshot 2025-08-13 113834.png)

After filtering out all of the images that could not be grouped together with at least one other image, this gave me a total of 3674 images. Even for finetuning standards, this is a relatively small dataset, but given the amount of time that it took to manually organize the dataset into groups, I was not willing to gather more data in this way at the moment and I just carried on with training the model:  
# Model Training
All of the experiments detailed below were run on either one NVIDIA A10 or one NVIDIA A100 using [Modal's](modal.com) cloud GPUs. For the GPU-poor like me, Modal is a pretty good option to run a few quick (moderate) experiments as they offer up to 30$ worth of free credits. 
Modal also makes it pretty easy to run the training code on their GPU's and allows to make quick changes without needing to scp or git push+pull a lot, as one would do on typical HPC clusters.
One really only needs to include the modal-setup and the function-wrapper 
```python
# Basic Setup (done once in code)
stub = modal.App("bike-search-finetuning-triplet")
volume = modal.Volume.from_name("bike-search-data")
image = (modal.Image.debian_slim()
    .pip_install([   # pip install necessary libraries
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.5.0",
    ])
    .pip_install("open_clip_torch")
    .add_local_file("./src/models/eval_search.py", remote_path="/root/eval_search.py")    # add file from local repository
)

# Function-wrapper for function to be run on Cloud-GPUs 
@stub.function(
    image=image,
    gpu="A100",
    timeout=18000,
    volumes={"/data": volume}
)
def train_model(...): 
	...
```
around the main-method and then one can run the training routine using the terminal command `modal run finetune.py`. Nothing else about the code needs to change.
To set up the dataset on Modal one needs to run `modal volume create bike-search-data` and transfer the training dataset via `modal volume put bike-search-data PATH_TO_DATASET [REMOTE_PATH]`. 
## Finetuning Embedding Models
For the embedding model, we will primarily focus on finetuning models of the [OpenCLIP family](https://github.com/mlfoundations/open_clip). More precisely, the OpenCLIP models are made up of a language encoder and a vision encoder; Here we will just use the image encoder for creating the embeddings. It would be a valid approach (and potentially something for future work) to train both the vision encoder using an image of the bicycle together with training the language encoder using e.g. a description of the bike or the title of the listing on the ecommerce platform.

We have also experimented with finetuning variants of the DinoV2 model, which often times gave comparable performance to the best results of the OpenCLIP models, but the DinoV2 model often times seemed to take longer to train and longer to calculate embeddings, thus making the DinoV2 variants a slightly worse option for us. 

### Training with the Triplet Loss 
A classic way to train embedding models for vector search is using the **Triplet Loss** function. The idea is as follows: We pass the model a set of 3 images, one "anchor" image, one "positive" image that is a different image of the same bicycle, and one "negative" image that is a different image of a completely different bicycle. Here are some examples of such anchor-positive-negative triplets: 
![](/images/Screenshot 2025-08-13 093944.png)![](/images/Screenshot 2025-08-13 094055.png)*(Left: Anchor Image; Middle: Positive Sample; Right: Negative Sample)*

The goal is of course that the embeddings of the "anchor" image and the "positive" image are similar, while simultaneously being very dissimilar from the embedding of the "negative" image. There are different ways of defining "similarity" of embeddings, the two most common ones being:
- euclidean distance, i.e. two embeddings $z_1,z_2\in\mathbb R^d$ are "similar" if their distance $\Vert z_1- z_2\Vert_2$ is close to $0$ and
- cosine similarity. i.e. two embeddings $z_1, z_2\in\mathbb R^d$ are "similar" if the cosine of the angle between the vectors $$\cos(\measuredangle(z_1,z_2))=\frac{\langle z_1,z_2\rangle}{\Vert z_1\Vert_2 \cdot \Vert z_2\Vert_2}\in[-1, 1]$$is close to $+1$. 
  
Thus if $x$ is our anchor image, $x^+$ our positive image and $x^-$ our negative image and letting $s(z_1,z_2)$ be the similarity score of two embeddings, we want $s(z,z^+)$ to be large and $s(z, z^-)$ to be small. We can achieve this by minimizing the value $s(z, z^-) - s(z, z^+)$. This is the main idea behind the triplet loss; which is defined by $$\mathcal L=\max \bigg(s(z, z^-) - s(z,z^+) + \alpha, 0\bigg)$$which essentially means the model learns to push $z$ and $z^-$ apart while keeping $z$ and $z^+$ close together *up until some threshold $\alpha>0$ had been reached*. The necessity of the "margin" parameter $\alpha$ can more easily be understood in the case of optimizing for euclidean distance: If we were to simply minimize the objective $\Vert z-z^+\Vert_2 - \Vert z-z^-\Vert_2$ the model would just learn to push apart $z$ and $z^-$ as far as possible, making the embeddings diverge (in the sense that $\Vert z\Vert_2\to\infty$) as training goes on. Thus this "thresholding" until reaching a difference of $\alpha$ can be understood as simply stabilizing training. On top of that, in terms of nearest-neighbour-retrieval there is minimal (or even no) benefit of seperating already well-seperated embeddings even further.

--------
Now that we have our loss function we can take a look at a few results of finetuning some ViTs of the OpenCLIP family with the triplet loss: 

1. **ViT-B-32:** We started out training one of the more light-weight models of the OpenCLIP family. The ViT-B-32 patches the input image into 32x32 patches. Due to this quite large patch-size and a fixed input resolution of 224x224, an input image will be divided into only $7\times 7$ tokens and hence the size of the subsequent attention matrices will be quite managable compared to the larger models of the OpenCLIP family. 
   The first experiment, taking AdamW as the optimizer with a moderate learning rate of $\text{lr}=10^{-3}$ or $\text{lr}=10^{-4}$ and setting the margin parameter $\alpha$ to $0.2$, we got the following intriguing result:    ![]((/images/Screenshot 2025-06-23 134746.png)
   What is happening here is that both train and validation Triplet Loss converge to $0.2$, meanwhile the retrieval metrics remain quite bad. This is a quite [well-known phenomenon](https://stats.stackexchange.com/a/475778/388069) when training with the triplet loss and there is a simple explanation of this behaviour: The embedding model simply learns to embed every input into the same constant embedding vector, i.e. the model collapses the entire embedding space into a single point. As a result, we will always have $\text{sim}(z,z^+)=\text{sim}(z, z^-)$ and thus $$\mathcal L=\max \bigg(s(z, z^-) - s(z,z^+) + \alpha, 0\bigg)=\alpha$$which in our case was exactly $0.2$, explaining the plateau of the losss at $0.2$. 
   What's commonly suggested to fix this issue is to implement "hard-negative mining", meaning we do not sample the negative example $x^-$ randomly from the entire dataset, but instead choose a specific negative anchor that we know the model currently finds difficult to distinguish from the anchor image $x$. While we did also implement this change, we surprisingly found that it did not have a large impact on fixing this problem in our case. However, what did solve the problem was simply tuning the learning rate to be much much smaller, going down to $\text{lr}=5\cdot 10^{-6}$, and increasing the margin parameter to $\alpha=0.4$; giving us the following results: ![](/images/Screenshot 2025-08-13 095240.png)
   *(Ignore the fact that the validation loss is way smaller than the train loss, this is just a consequence of using hard-negative mining for the train loss but not for the val loss. We mainly care about the validation Recall anyway, so the loss value is not so important here ...)*
   A Recall@10 of $\approx 57.5\%$  is not bad at all, considering that for each anchor image there are typically only $1$ to $3$ similar images to be retrieved out of a dataset of thousands of images! 
   Another interesting observation is that the model reaches a good recall quite quickly and then does not seem to improve even as training goes on for a long time. This leveling-off of the Triplet-loss curve is again relatively common and it occurs when the typical triplets $(x, x^+, x^-)$ are quite easy for the model to cluster appropriately, seperating $x$ and $x^-$ while keeping $x$ and $x^+$ close. Hard-negative mining is again supposed to help with this issue, giving the model more difficult triplets to distinguish, thus requiring the model to learn more novel features that are necessary to handle more difficult samples. Again, we did implement hard-negative mining but found virtually no effect on the training results, again quickly reaching the same Recall metrics, then plateauing.
   At this point I thought that the coarse patching of the ViT-B-32 might be a bottleneck for learning more novel features. For example, many bike frames have the name of the brand and sometimes even the name of the model written on the bike frame; In this case, given the already small resolution of $224\times 224$ a patch size of $32\times 32$ might be too coarse to extract this feature. Thus we tried out models with larger patch sizes:
2. **ViT-B-16:** 
   The ViT-B-16 model has exactly the same architecture as the ViT-B-32 model and also requires a fixed input resolution of $224\times 224$. The only difference is the patch-size now being $16\times 16$ instead of $32\times 32$, leading to $4\times$ the amount of tokens per image, which in turn leads to the size of the attention matrices being $16\times$ larger compared to the ViT-B-32 model. As a consequence, we did require better hardware for training this model (switching from an A10 GPU to an A100 GPU). Luckily this is an easy one-line change when using Modal. 
   This model change, keeping all hyperparameters the same, got us the following results: ![](/images/Screenshot 2025-08-13 101746.png)
   We do see the same phenomenon of quickly reaching good recall metrics, then barely improving over the following epochs. Nevertheless, after 10 epochs we get to $\text{Recall@}10 = 65\%$, which is a quite significant improvement from our ViT-B-32 results! 
   
   At this point you might also rightly ask if the model is even learning anything at all or if the base model already achieves these metrics without any finetuning: The model is indeed learning a lot very quickly during the first epoch and the base model is much worse. We also trained the same model with SGD, which only achieved a $\text{Recall@}10$ of $~15\%$ after the first epoch, then improved consistently during the following epochs, but eventually reached roughly the same level of $\text{Recall@}10\approx 60\%$ and then plateaued. Therefore (a) the model is indeed learning something and (b) AdamW makes the model learn it much faster than SGD.
   
3. **ViT-L-14:** 
   The ViT-L-14 not only has an even smaller patch size of $14\times 14$, but the base model itself is also much larger, having $24$ Transformer layers compared to the $12$ layers that the ViT-B-x models have (the ViT-L model also has more attention heads per layer and a larger hidden dimension). Training with this larger model gave some additional improvements, already getting $\text{Recall@}10=44.8\%$ after the first epoch of training. However, at this point we can see that switching to a larger model gives marginal returns and the computational requirements for this model are too large to run it on a moderate CPU in real time. Thus we ultimately decided to stick with the ViT-B-16 model.   

In short, we have a model (ViT-B-16) that get's a $\text{Recall@}10$ of approximately 65% while still being performant enough to run on a moderate CPU (t3.medium) in real time. 
The main bottlenecks to achieving better retrieval metrics very likely are a mixture of: 
1. The training dataset being too small
2. The input resolution being too low
3. The model being too small
While points 2 and 3 are in theory easily fixable, they are at odds with us wanting a highly performant model that still runs quite quickly on a moderate CPU. 
However, beyond finetuning the Embedding model, there are a few different improvements one can make to a vector search pipeline, which we will go into a bit now:
## Further Improvements for Vector Search 
### Training a Reranking Model 
Another option to improve the retrieval pipeline after the embedding model has been trained is to train a so-called **Reranking Model** or **Reranker**. The basic idea is as follows: Given the trained embedding model and a query image, we calculate the query embedding and perform the nearest-neighbour search in our database, giving us some amount of search results, let's say $64$. Now since our $\text{Recall@}10$ is way below $100\%$, it may very well be possible that there are some relevant results that are in the top-64 but not in the top-10. Thus if we can train a model that re-arranges the top-64 such that the most relevant items are indeed in the top-10, this would greatly improve our Recall@10 without changing our embedding model at all! 

But how would that be possible? Isn't the embedding model already trained to put all of the relevant results into the top-$k$? If our embedding model can't do it, why should a "Reranking model" be able to do it?

The main point which makes this idea viable is the following: The embedding model never actually "compares" two samples (query, result). Instead, it computes features/embeddings for each sample independently, then the vector database simply compares these features to retrieve the most similar images. I.e. an embedding model is trained on inputs containing just one sample. 
A reranker on the other hand is usually trained using tuples of images, i.e. one query image and one result-image from the top-$k$. This allows the model to directly compare the two data points and does not treat them independently. 
Therefore, one can train a model on pairs $(x_{\text{query}}, x_{\text{result}_i})$ with labels $y_i=1$ if $x_{\text{result}_i}$ is indeed a relevant result and $y_i=0$ otherwise, then take the output of this classification model as a similarity score and re-order the top-$k$ search results by sorting by the similarity scores of the reranker.

We did experiment a bit using some simple models for the reranker, but were not able to get any significant improvements so far. At this point it may also be possible to introduce a multi-modal model for training the reranker, as the title/description of the results often times is very helpful in deciding whether a search result is relevant, e.g. by checking the name of the brand. Though this would require (a) the user to enter the brand name of their stolen bike, which is sometimes unknown to bike owners and (b) the title/description of the listing to be meaningful and contain important keywords, which is not always the case (sometimes the title is just something like "Mountainbike new"). 
Experimenting a bit more with reranking models is definitely on the To-Do list for future work, but currently the search engine works without a reranker.  

### Adding Filters 
Since we are looking for stolen bicycles on ecommerce platforms, it makes a lot of sense that if a user searches for his bike that was stolen on, say, the 1st of July, that we only display search results that were postes online on or after the 1st of July. 
Of course, this can be done by simply filtering the search results after the vector database has been queried, but this might reduce the amount of search results drastically. For example, say we query the top 60 results from the database, then filter by our date requirement. Then it may happen that a large portion of the top-60 get filtered out and the user might be left with only a few search results. 
Luckily, some vector databases allow to make queries to the vector database with extra filters, such that the filter is applied before/during the nearest-neighbour search, ensuring that we always get as many search results satisfying our requirements as we query. For ChromaDB, which we are currently using, this looks as follows: 
```python 
results = chroma_collection.query(
	query_embeddings=[query_emb.tolist()],
	n_results=60,
	include=["metadatas", "distances", "embeddings"],
	where={"posted_timestamp": {"$gte": cutoff_timestamp}} # this is the date-filter
)
```
This of course requires to store the attribute "posted_timestamp" as metadata in the vector DB. 
It is also worth noting that we did not perceive any significant latency increase for the vector DB queries after implementing this additional filter.   

On top of that, we might also include a keyword search, e.g. for the name of the brand of the bike, in the search, meaning we only display the nearest-neighbours whose title/description include the keyword. Unfortunately, ChromaDB does not offer a method to include this keyword search within the query to the chroma collection, thus we can only filter the search results *after* they have been returned by the database, leading us to the same problem as described before that the amount of search results will be greatly reduced, possibly even become 0.  

In short, we have the following two options: 
1. Include the filter into the vector DB query itself, ensuring that we always get the exact amount of results we queried. Though this is only implemented for certain types of filters, e.g. numerical filters of the form "where $\text{feature}\geq x$"
2. Implement the filter *after* the query to the vector DB has been performed and the top-k have been returned. This offers a greater flexibility of allowing us to use pretty much any filter we desire, without requiring ChromaDB to have implemented a specific method for it, but greatly reduces the amount of search results or might even remove all search results. 
## Summary

**Main Findings:**
1. Increasing input resolution and/or making the patch size smaller (by switching from ViT-B-32 to ViT-B-16) immediately increases Recall@10 by about 10% (going from the previously best 57% to 65%). Though this does slow down training and inference noticably. 
2. Triplet Loss gave the best results (in terms of retrieval metrics like Recall@10) among all of the contrastive losses I tried; Though results varied a lot depending on Hyperparameters of the loss function and optimizer.
3. Increasing the dataset size by adding synthetic data didn't do much.

**Other stuff we tried, but didn't really achieve any improvements with:** 
1. Due to the lack of training data, I tried generating a synthetic dataset by taking images of bicycles with white backgrounds, then augmenting the background using commonly seen backgrounds on natural images (e.g. a garage, a street or a lawn) and rotating / mirroring the bike. This did greatly increase the amount of training samples, but training with the original dataset plus the synthetic dataset barely improved the retrieval metrics (we're talking about getting at most +1% on Recall@10, which might just be coincidental...).
2. Some might say that the Triplet Loss has somewhat gone out of fashion and these days people typically train such embedding models using contrastive Losses like the [InfoNCE Loss](https://arxiv.org/pdf/1807.03748). I did try training with some other common loss functions for embedding models, but while some loss functions came close to getting the retrieval performance I got using the Triplet Loss, none beat the Triplet Loss. Though admittedly I did commit to using the Triplet Loss relatively early on, thus I have done a lot more experiments with the Triplet Loss than, say, the InfoNCE Loss. Therefore it may very well be possible that there is some improvement to be found when properly experimenting with other contrastive loss functions. 

# Deployment & Some Final Inference Runtime Optimizations
Finally, we want to deploy the model on our EC2 (t3.medium) instance that does not have a GPU available. While the ViT-B model is not exactly the largest model around, it's not small either so we can quickly run into issues (either OOM-errors or a runtime that's too long) if we are not careful. 
The database get's updated with ~50 new items every minute, thus we need our model to easily handle this amount of data in the given timeframe.

Avoiding OOM errors is relatively easy by just decreasing the batch size. On a CPU, this doesn't even increase the runtime of the inference at all, at least not for the values I tried. 

Improving runtime is more tricky unfortunately. While there are some easy tricks one should definitely do, like using `torch.compile` which is just a one-line change in the code, they typically did not change runtime a lot in my experiments. 
The biggest difference was introducing dynamic qint8 [Quantization](https://docs.pytorch.org/docs/stable/quantization.html) of the MLPs in the transformer blocks, but quantization changes the model weights and thus changes the output of the embedding model. After running some experiments evaluating the retrieval metrics of the quantized model, I quickly realized that there is a significant performance drop when using the quanitzed model compared to the base model:
```
**Metrics without Int8-Quantization:** 
Recall@1: 0.2212 | Precision@1: 0.5264
Recall@5: 0.4723 | Precision@5: 0.3008
Recall@10: 0.6480 | Precision@10: 0.2232

**Metrics with qInt8-Quantization:**
Recall@1: 0.1847 | Precision@1: 0.4396
Recall@5: 0.4035 | Precision@5: 0.2581
Recall@10: 0.5262 | Precision@10: 0.1924
```

The moderate improvement in runtime is not enough to justify this strong drop in performance, thus I ultimately chose to drop the quantization and keep the base model. 
The base model is fast enough to handle the amount of new data each minute most of the time anyway.
In theory, one could re-train the embedding model using [Quantization-Aware-Training](https://pytorch.org/blog/quantization-aware-training/) to ease the perfomance drop a bit; But I've put this further down on the To-Do list for now ...

---- 
**That's it, hope you enjoyed :)** 




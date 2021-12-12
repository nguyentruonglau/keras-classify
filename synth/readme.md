# SYNTH DATA

Generate random data: for each randomly selected image, 
that image will be augmented randomly, saved with the structure `<aug>_<random number>_<current time>_<basename file>`

Add more augmentation [here](https://github.com/nguyentruonglau/aic_classify/blob/main/synth/utils.py)

## I. DIRECTORY STRUCTURE:

```
	input:
			--image_1.jpg
			--image_2.png
			--image_3.jpeg
			...
	output expect:
			--aug_[random number]_[time]_image_1.jpg
			--aug_[random number]_[time]_image_2.png
			--aug_[random number]_[time]_image_3.jpeg
			... 				
```

## II. SYNTH DATA:

```
>> python synth_data.py
```

## IV. CONTACT:
ttruongllau@gmail.com

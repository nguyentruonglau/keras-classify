# CONVERT TO NPY FILES 

Read the datas and save it as npy files


## I. DIRECTORY STRUCTURE

### A. custom_split_data==False
(auto split train-val-test)

```
				input_dir -- class_one_dir
		        			 --image_1.jpg
		        			 --image_2.png
		        		  -- class_two_dir
		        		        --image_1.jpeg
		        		        --image_2.png
		        		  -- ..
```

### B. custom_split_data==True
(Custom plit train-val-test)

```
		        input_dir -- train
		        		--class_one
			        		--image_1.jpg
			        		--image_2.png
				        --class_two
				        	--image_1.jpeg
				        	--image_2.png
		        		-- ..
		        	  -- val
		        		--class_one
			        		--image_1.jpg
			        		--image_2.png
				        --class_two
				        	--image_1.jpeg
				        	--image_2.png
		        		-- ..
		                 -- test
		        		--class_one
			        		--image_1.jpg
			        		--image_2.png
				        --class_two
				        	--image_1.jpeg
				        	--image_2.png
		        		-- ..        		  
```

## II. CONVERT DATA:

```
>> python convert_npy.py
```

## IV. CONTACT:
ttruongllau@gmail.com

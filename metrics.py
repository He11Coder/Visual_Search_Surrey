import numpy as np
import config

def computePR(query_image, output_images):
    pr_stat = np.empty(len(output_images), dtype="f,f")

    query_class = int(query_image.split('_')[0])
    tr_pos_count = 0
    for i in range(len(output_images)):
        img_filename = output_images[i]
 
        if query_class == int(img_filename.split('_')[0]):
            tr_pos_count += 1
        
        pr_stat[i] = (tr_pos_count/config.NUM_OF_ELS_IN_CLASS[query_class-1], tr_pos_count/(i+1))
    
    return pr_stat
import numpy as np
from operator import mul
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
from components import feeders
import random
import matplotlib.pyplot as plt
from random import sample
from PIL import Image
from PIL import ImageDraw
from scipy import ndimage

## n = 2
## k = 2~5

class fat_chain(feeders.Feeder):

    def initialize_vars(self,
                        item_size, box_extent,
                        raw_input_size=None,
                        n=2, num_item_pixel_values=2,
                        organization = 'raw',
                        display=False):

        if raw_input_size is not None:
            self.raw_input_size = raw_input_size

        self.organization = organization #can be 'full', 'obj', 'raw'
        self.item_size = item_size
        if len(item_size) == 2:
            self.item_size += [self.raw_input_size[2]]

        self.box_extent = box_extent
        self.num_item_pixel_values = num_item_pixel_values
        self.display = display
        self.k = k
        self.n = n

        if self.organization != 'raw':
            self.actual_input_size = self.raw_input_size[0:2] + [self.n*self.k]
        else:
            self.actual_input_size = self.raw_input_size



    def single_batch(self, label_batch=None):
        # 1.
        #   if sd_portion = 0.5, patch = 2x2, # different pixels = 2,
        #   it's categorized as 'SAME', POSITIVE
        # 2.
        #   if sp_portion = 1 and the two patches lie along a diagonal,
        #   it's categorized as 'UD', POSITIVE

        input_data = np.zeros(dtype=np.float32, shape=(self.batch_size,) + tuple(self.actual_input_size))
        target_output = np.zeros(dtype=np.float32, shape=(self.batch_size, 1, 1, 2))
        if label_batch is None:
            label_batch = np.random.randint(low=0, high=2, size=(self.batch_size))
        elif label_batch.shape[0] != (self.batch_size):
            raise ValueError('label_batch is not correctly batch sized.')

        positions_list_batch = []
        items_list_batch = []
#        if self.num_items != 2:
#            raise ValueError('Num items other than 2 not implemented yet.')

        iimage = 0
        while iimage < self.batch_size:
            positions_list = []
            items_list = []
            # sample positions
            positions_list = sample_positions_naive(self.box_extent, self.item_size, self.n*self.k, list_existing=positions_list)

            if label_batch[iimage] == 0: # Negative
                # sample bitpatterns
                items_list = sample_bitpatterns_naive(self.item_size, self.n*self.k,
                                                      self.num_item_pixel_values, list_existing=items_list,
                                                      force_different=True)
                for group in range(self.n-1): # other negative cases by chance
                    for item in range(self.k-1):
                        if np.random.randint(low=0, high=2) == 1: # current item is 'same' as the previous item in group
                            items_list[group*self.k+item] = items_list[(group+1)*self.k-1]
            else:                        # Positive
                for group in range(self.n):
                    # sample bitpatterns
                    items_list = sample_bitpatterns_naive(self.item_size, len(items_list) + 1,
                                                          self.num_item_pixel_values, list_existing=items_list,
                                                          force_different=True)
                    for extra_copy in range(self.k - 1):
                        items_list.append(items_list[-1])

            # render
            image = self.render(items_list, positions_list, label_batch[iimage])
            target_output[iimage, 0, 0, label_batch[iimage]] = 1
            input_data[iimage, :, :, :] = image
            iimage+=1
            if self.display:
                print(target_output[iimage-1,0,0,:])
                positions_list_batch.append(positions_list)
                items_list_batch.append(items_list)

        return input_data, target_output, positions_list_batch, items_list_batch


    def render(self, items_list, positions_list, label):

        if len(items_list) != len(positions_list):
            raise ValueError('Should provide the same number of hard-coded items and positions')


def generate_image(n_obj):
    size = [200,200]
    max_rad = 30 #20
    total_length = 160
    dilation = 20
    sketchboard = np.zeros((size[0],size[1]))
    old_dilated = np.zeros((size[0], size[1]))
    for i in range(n_obj):
        failed = True
        while failed:
            obj, length, junctions = get_object(size, max_rad, total_length, dilation, display_process=False)
            translated = random_translate(obj, size) #randomly translate
            if translated is None:
                failed = True
                continue
            else:
                failed = False
            outline, dilated = get_outline(translated) #get outline
            old_dilated = np.maximum(dilated, old_dilated)
            sketchboard = sketchboard*(1-dilated) + outline*(dilated)
    return sketchboard


def random_translate(obj_img, size):
    vertical = list(np.amax(obj_img,axis=1))
    horizontal = list(np.amax(obj_img,axis=0))

    top_slack = -np.argmax(vertical)
    bottom_slack = np.argmax(list(reversed(vertical)))
    left_slack = -np.argmax(horizontal)
    right_slack = np.argmax(list(reversed(horizontal)))

    if ((bottom_slack - top_slack) < size[0])| \
        ((right_slack - left_slack) < size[1]):
        print('object too big. resampling.')
        return None

    else:
        aligned = np.roll(np.roll(obj_img,top_slack,axis=0),left_slack,axis=1)
        aligned = aligned[:size[0],:size[1]]
        vertical = list(np.amax(aligned, axis=1))
        horizontal = list(np.amax(aligned, axis=0))
        bottom_slack = np.argmax(list(reversed(vertical)))
        right_slack = np.argmax(list(reversed(horizontal)))
        vertical_movement = np.random.randint(0,bottom_slack+1)
        horizontal_movement = np.random.randint(0,right_slack+1)
        return np.roll(np.roll(aligned,vertical_movement,axis=0),horizontal_movement,axis=1)

def get_outline(obj_img):
    struct1 = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(obj_img, structure=struct1, iterations=int(1)).astype(float)
    return dilated - obj_img, dilated

def get_object(size, max_rad, total_length, dilation, display_process=False):
    if total_length <= max_rad:
        return ValueError('total_length should be greater than step size(max_rad)')
    if float(dilation)*4/3 > float(max_rad):
        print('WARNING: dilation is too large compared to chain step size. Too many lines will be rejected from sampling.')
    if max_rad >= 2*dilation:
        print('WARNING: step size is too large compared to object thickness. Object might self-cross (creating loops).')

    head = [size[0]*2/2,size[1]*2/2]

    # initialize mask
    mask = Image.new('F', (size[0]*2,size[1]*2), 'white')
    draw = ImageDraw.Draw(mask)
    draw.ellipse((dilation,dilation, size[0]*2-dilation, size[1]*2-dilation), fill='black', outline='black')
    mask = np.array(mask)/255

    current_length = 0.
    num_junctions = 0

    while current_length < total_length:
        rad = np.maximum(np.maximum(max_rad*np.exp(-(np.random.rand())**2),float(max_rad)/3), float(dilation)*2/3)
        incremental_im, incremental_mask, head = get_line(head, rad, mask, dilation)
        if incremental_im is None:
            print('dead end')
            break
        if num_junctions == 0:
            im = incremental_im

        im, mask = update(im, mask, incremental_im, incremental_mask)
        current_length += rad
        num_junctions += 1

        if display_process:
            plt.subplot(1,2,1)
            plt.imshow(im)
            plt.subplot(1, 2, 2)
            plt.imshow(mask)
            plt.plot(head[1],head[0],'ro')
            plt.colorbar()
            plt.show()

    #final dilation 50%
    blurred = ndimage.filters.gaussian_filter(im, sigma=float(dilation)*1/4, mode='reflect', cval=0.0, truncate=3.0)-0.5
    blurred2 = (blurred > 0).astype(float)

    if display_process:
        plt.imshow(blurred2)
        plt.show()
    return np.array(blurred2), current_length, num_junctions


def update(old_im, old_mask, incremental_im, incremental_mask):
    new_im = np.maximum(old_im, incremental_im)
    new_mask = np.maximum(old_mask, incremental_mask)

    return new_im, new_mask

# given the coordinate (y,x) of the center and a fixed radius and a mask,
# returns an image of a line, and a corresponding 'drawability mask'
def get_line(head, rad, mask, dilation_rad):
    # get the list of pixels to sample from
    uniques, cmf, pmf = get_circle(head, rad, mask)
    if pmf is None:
        return None, None, None
    random_num = np.random.rand()
    sampled_ind = np.argmax(cmf-random_num > 0)
    tail = uniques[sampled_ind, :] # find the smallest index whose value is greater than rand

    # IMAGE
    # draw a line with 25% of specified thickness
    lin = Image.new('F', (mask.shape[0], mask.shape[1]), 'black')
    draw = ImageDraw.Draw(lin)
    draw.line([tuple(head),tuple(tail)], fill='white', width=int(float(dilation_rad)*1/4))

    # dilate with additional 25% of specified thickness (for smoothness)
    struct1 = ndimage.generate_binary_structure(2, 1)
    im = ndimage.binary_dilation(lin, structure=struct1, iterations=int(float(dilation_rad)*1/4)).astype(float)

    # MASK
    # compute mask using max integration
    lin2 = Image.new('F', (mask.shape[0], mask.shape[1]), 'black')
    draw2 = ImageDraw.Draw(lin2)
    draw2.line([tuple(head),tuple(tail)], fill='white', width=int(float(dilation_rad)*1.5))
    lin2 = ndimage.binary_dilation(lin2, structure=struct1, iterations=int(float(dilation_rad)*3/5)).astype(float)
    blurred = ndimage.filters.gaussian_filter(lin2, sigma=dilation_rad*1.5, mode='reflect', cval=0.0, truncate=3.0)
    blurred = blurred/np.max(blurred)
    new_mask = np.maximum(np.array(lin2), blurred/2)

    return np.transpose(im), np.transpose(new_mask), tail

# given the coordinate (y,x) of the center and a fixed radius and a mask,
# returns a list of coordinates that are nearest neighbors around a circle cetnered at (y,x) and of radius rad.
# Also returns the CMF at each of the returned coordinates.
def get_circle(center, rad, mask):

    height = mask.shape[0]
    width = mask.shape[1]

    # compute angle of an arc whose length equals to the size of one pixel
    deg_per_pixel = 360./(2*rad*np.pi)
    samples_in_rad = np.arange(0,360,deg_per_pixel)*np.pi/180

    # get lists of y and x coordinates (exclude out-of-bound coordinates)
    samples_in_y = np.maximum(np.minimum(center[0]+(rad*np.sin(samples_in_rad)), height-1), 0)
    samples_in_x = np.maximum(np.minimum(center[1]+(rad*np.cos(samples_in_rad)), width-1), 0)
    samples_in_coord = np.concatenate((np.expand_dims(samples_in_y, axis=1),
                                       np.expand_dims(samples_in_x, axis=1)), axis=1).astype(int)

    # find unique coordinates and prune
    uniques = np.unique(samples_in_coord,axis=0)
    inverted = 1-mask[uniques[:,0],uniques[:,1]]
    pmf = np.array([inverted])
    total = np.sum(pmf)
    if total < 0.05:
        return uniques, None, None

    pmf = pmf/total
    cmf = np.cumsum(pmf)

    return uniques, cmf, pmf

if __name__ == '__main__':
    #get_object(size= [120,240], max_rad=10, total_length=70, dilation=7, display_process=False)
    n_counts = sys.argv[1]
    n_images_per_count = sys.argv[2]
    counts = range(1,n_counts+1)
    batch_size = sys.argv[3]
    curr_batch = []
    for i in n_counts*n_images_per_count:
        curr_count = sample(counts,1)[0]
        image = generate_image(curr_count)

    generate_image()

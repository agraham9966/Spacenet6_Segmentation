import numpy as np 
import sys, os 
import matplotlib.pyplot as plt 
import gdal, ogr, osr 
import osgeo.gdalnumeric as gdn
import re 


def json_to_mask(img_path, vector_path, burn_val=1): 

    '''
    :params img_path: path of raster corresponding to annotation shapefile
    :params vector_path: path of vector containing raster annotations (each class should be an attribute)
    '''

    # only classifying buildings in spacenet 6 
    bands=1 

    #open raster layer 
    raster_src = gdal.Open(img_path, gdal.GA_ReadOnly)
    xSize, ySize = raster_src.RasterXSize, raster_src.RasterYSize
    geotransform = raster_src.GetGeoTransform()
    spatialref = raster_src.GetProjection()

    #open vector layer with JSON driver 
    driver = ogr.GetDriverByName('GeoJSON')
    dataSource = driver.Open(vector_path, 0) # 0 means read-only. 1 means writeable.
    layer = dataSource.GetLayer()

    if dataSource is None: 
        print('Could not open %s' % vector_path)
        exit()
    else: 
        print('Opening Shapefile...')

    #set up raster driver from memory 
    target_ds = gdal.GetDriverByName('MEM').Create(
        '', 
        xSize, 
        ySize, 
        bands, 
        gdal.GDT_Byte
    )

    target_ds.SetGeoTransform(geotransform)
    target_ds.SetProjection(spatialref)

    #rasterize shapefile, put each attribute in shapefile into its own band 
    gdal.RasterizeLayer(
        target_ds, 
        [1], 
        layer, 
        options = ["ALL_TOUCHED=False", "ATTRIBUTE=Building_ID"], 
        burn_values=[burn_val] #not working for some reason 
    )

    shapemask_array = target_ds.ReadAsArray().astype('float32') #keeping same datatype as input scene because of merging during tile process
    
    #make sure one-band masks are shaped (bands, rows, cols)
    if len(shapemask_array.shape) == 2: 
        shapemask_array = np.expand_dims(shapemask_array, axis=0)

    #temp solution to overcome gdal rasterize layer bug... ensure all binary values are 0,1
    for band in range(shapemask_array.shape[0]): 
        shapemask_array[band][shapemask_array[band]>0] = 1.

    target_ds=None 

    return shapemask_array

def geotrans_upscale(geotransform, og_height, og_width, rows_padded, cols_padded): 

    ##if padding is added to an image to generate chips, the upper left x,y coords are updated from 
    #gdals original geotransform output 
    #returns updated geotransform tuple 

    ul_x, ul_y = geotransform[0], geotransform[3]
    pix_width, pix_height = geotransform[1], geotransform[5]

    ul_y = ul_y-(rows_padded*pix_height)
    ul_x = ul_x-(cols_padded*pix_width)

    return (ul_x, geotransform[1], geotransform[2], ul_y, geotransform[4], geotransform[5])

def img_chipsizer(img, wind_size=16): 
    
    '''
    :params img_path: path to geo-tiff or envi format image with spatial reference 
    :params wind_size: size of image chips you wish to generate
    
    Resizes image based on desired chip/window size. If your window size doesn't tile 
    equally along rows and/or columns, padding is added to the rows and/or columns. 

    ''' 

    #img, geo, spa = img_to_array(img_path, dtype=img_dtype) #shape=(bands, rows, cols)

    num_bands, num_rows_og, num_cols_og = img.shape[0], img.shape[1], img.shape[2]

    if num_rows_og % wind_size == 0: 
        num_rows = num_rows_og
        rows_padded = 0
    else: 
        num_rows = int(((num_rows_og//wind_size)+1)*wind_size)
        rows_padded = (num_rows - num_rows_og) //2 #num rows to pad from left portion of image 

    if num_cols_og % wind_size == 0: 
        num_cols = num_cols_og
        cols_padded = 0
    else: 
        num_cols = int(((num_cols_og//wind_size)+1)*wind_size)  
        cols_padded = (num_cols - num_cols_og) //2 #num cols to pad from left portion of image 

    print('#Tiles: {} rows x {} cols'.format(int(num_rows/wind_size), int(num_cols/wind_size)))
    num_tiles = int((num_rows/wind_size)*(num_cols/wind_size))

    if img.shape != (num_bands, num_rows, num_cols): 
        
        print('original image size={}, final image size ={}'.format(img.shape, (num_bands, num_rows, num_cols)))
        print('resizing....') 
        
        padded = np.zeros((num_bands, num_rows, num_cols))
        padded[:, rows_padded:num_rows_og+rows_padded, cols_padded:num_cols_og+cols_padded] = img
        
        return padded, num_tiles

    else: 
        print('no image resizing necessary, returning original image and spatial info')
        return img, num_tiles

def Xy_chipgen(X_img, y_img, wind_size=320, remove_bad_tiles=True, mask_zeros=True): 

    '''
    :param img_path: path of raster corresponding to annotation shapefile
    :param vector_path: path of vector containing raster annotations (each class should be an attribute)
    :param dn_max: what value to normalize the image by, or bit depth of remote sensing image 
    :param img_dtype: how to open image in img_path - by default opens as float 32 to perform normalization (0-1)  
    :param wind_size: row and colsize of window tiles in pixels 
    :param num_attclasses: number of classes or att.fielfd in the shapefile (code will stop if there are too many fields)
    :param remove_bad_tiles: removes X and y tiles from input image (img_path) which contain all zero vals in X tiles
    :param mask_zeros: masks y_tile values if edge cases occur (if encoded mask values overlap with no_data region in image) 

    outputs X_tiles, y_tiles arrays in shape=(#tiles, rows, cols, bands)
    '''

    M,N = wind_size, wind_size #explicitly defining rows and cols size is all 

    #stack mask with input image 
    print('Merging input image with mask...')
    comb_array = np.vstack((X_img, y_img)) #stacks along axis 0 (bands)
    
    #resize and add padding if padding is needed 
    padded, ntiles = img_chipsizer(
        comb_array,  
        wind_size=wind_size
    )
    
    #shaped each tile as channels last... or figure out how to set channels first in tensorflow 
    X_tiles = [padded[:X_img.shape[0], x:x+M,y:y+N].transpose((1, 2, 0))
            for x in range(0,X_img.shape[1],M) 
            for y in range(0,X_img.shape[2],N)] 
    X_tiles = np.array(X_tiles)

    # y_tiles = [padded[X_img.shape[0]:, x:x+M,y:y+N].transpose((1, 2, 0)).astype('byte') 
    #         for x in range(0,X_img.shape[1],M) 
    #         for y in range(0,X_img.shape[2],N)] 

    y_tiles = [padded[X_img.shape[0]:, x:x+M,y:y+N].transpose((1, 2, 0))
            for x in range(0,X_img.shape[1],M) 
            for y in range(0,X_img.shape[2],N)] 
    
    y_tiles = np.array(y_tiles)

    #remove X_train and corresponding y_train tiles if 
    #X_train tiles are filled with 0 values across all bands  
    #since some images have huge no_data regions on edges 
    bad_tile_idx = []

    if remove_bad_tiles == True: 
        for i, tile in enumerate(X_tiles): 
            if np.all(tile==0) == True: 
                bad_tile_idx.append(i)

        if len(bad_tile_idx) >0: 
            print('Removing {} bad tiles'.format(len(bad_tile_idx)))
            X_tiles = np.delete(X_tiles, bad_tile_idx, axis=0)
            y_tiles = np.delete(y_tiles, bad_tile_idx, axis=0)

        else: 
            print('No bad tiles found.')

    # #mask out no_data regions from y_masks
    if mask_zeros == True: 
        for i, tile in enumerate(X_tiles): 
            mask = np.expand_dims(np.any(tile>0, axis=2).astype('byte'), axis=2)
            y_tiles[i] = y_tiles[i] * mask 

    return X_tiles, y_tiles

def img_to_array(input_file, dtype='uint16'):   ###reads multiband image as ndarray
    #specify band list - indices of each band# you want to open as multiband array 
    #not specifying band list opens all bands into an array 
    #returns wavelengths only when header is available for image 

    file  = gdal.Open(input_file)
    bands = [file.GetRasterBand(i) for i in range(1, file.RasterCount + 1)]
    arr = np.array([gdn.BandReadAsArray(band) for band in bands]).astype(dtype) 
    #geotransform = file.GetGeoTransform()
    #spatialreference = file.GetProjection()

    #sometimes single band images have shape of length 2- reshape to (bands, rows, cols) to be consistent
    if len(arr.shape) == 2:  
        arr = np.reshape(arr, (1, arr.shape[0], arr.shape[1]))
    
    return arr


def sort_spacenet_images(X_train_opt_path, X_train_sar_path, y_train_path): 

    #example image name format from spacenet download: SN3_roads_train_AOI_3_Paris_MS_img6.tif 
    #returns a list, where each row represents matching optical image path at element 0, sar at element 1, labels path at element2

    X_train_opt_ids = os.listdir(X_train_opt_path)
    X_train_sar_ids = os.listdir(X_train_sar_path)
    y_train_ids = os.listdir(y_train_path)

    #deal with uneven amount of training data and training labels - match up by image number
    Xy_train_names = [] #X_train, y_train 
    for i in X_train_opt_ids: 
        im_num = re.search('tile_(.*).tif', i)
        im_num = im_num.group(1)
        for j in y_train_ids: 
            _im_num = re.search('tile_(.*).geojson', j)
            _im_num = _im_num.group(1)
            for k in X_train_sar_ids:
                im_num_ = re.search('tile_(.*).tif', i)
                im_num_ = im_num_.group(1)
                if im_num == _im_num == im_num_: 
                    #print(im_num, _im_num)
                    Xy_train_names.append([X_train_opt_path + os.sep + i, 
                                           X_train_sar_path + os.sep + k,
                                           y_train_path + os.sep + j])
                break 

    print('{} Xy Matching Training images Found'.format(len(Xy_train_names)))

    return Xy_train_names #list of image paths and their mask paths 

def tilestack_opt_and_sar(): 

    #create one dataset which is optical and sar , no y_labels needed for GAN network 
    #create one dataset which is sar and y_labels 
    wind_size = 256 #size of chips for training 
    X_train_opt_path = 'D:/project_data/spacenet6/SN6_buildings_AOI_11_Rotterdam_train_sample/AOI_11_Rotterdam/PS-RGBNIR'
    X_train_sar_path = 'D:/project_data/spacenet6/SN6_buildings_AOI_11_Rotterdam_train_sample/AOI_11_Rotterdam/SAR-Intensity'
    y_train_path = 'D:/project_data/spacenet6/SN6_buildings_AOI_11_Rotterdam_train_sample/AOI_11_Rotterdam/geojson_buildings'
    Xy_train_names = sort_spacenet_images(X_train_opt_path, X_train_sar_path, y_train_path)

    #pair optical with sar and make tiles 
    X_train_opt = []
    X_train_sar = []

    for i in Xy_train_names: 

        img_opt = img_to_array(i[0], dtype='float32') / 2**11
        img_sar = img_to_array(i[1], dtype='float32') / 100
        y_mask = json_to_mask(i[0], i[2]) 

        #deal with black bars in imagery, gets bounds of valid data across all bands 
        idx_bounds = np.where(np.any(img_opt, axis=0).astype('byte') != 0)

        #clip to bounds 
        img_opt_clip = img_opt[:, idx_bounds[0].min(): idx_bounds[0].max() + 1, idx_bounds[1].min(): idx_bounds[1].max()+1]
        img_sar_clip = img_sar[:, idx_bounds[0].min(): idx_bounds[0].max() + 1, idx_bounds[1].min(): idx_bounds[1].max()+1]
        y_mask_clip = y_mask[:, idx_bounds[0].min(): idx_bounds[0].max() + 1, idx_bounds[1].min(): idx_bounds[1].max()+1]

        print(img_opt_clip.shape)

    #     #split images into tiles, add padding 
    #     _X_train_opt, _X_train_sar = Xy_chipgen(img_opt_clip, img_sar_clip, wind_size=wind_size, mask_zeros=False)

    #     X_train_opt.append(_X_train_opt)
    #     X_train_sar.append(_X_train_sar)
    
    # X_train_opt = np.concatenate(X_train_opt)
    # X_train_sar = np.concatenate(X_train_sar)

    # np.save('X_optical_samp.npy', X_train_opt)
    # np.save('X_sar_samp.npy', X_train_sar)

    return 


if __name__ == '__main__': 
    tilestack_opt_and_sar()

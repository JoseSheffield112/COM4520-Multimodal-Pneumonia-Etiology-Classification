import torch

class darwin_multimodal_dataset(torch.utils.data.Dataset):
    '''
    Dataset class used in dataloaders for the MMDL model. Mostly used to transform and scale images 
    from the image modality every time they're accessed to help avoid overfitting. 
    '''
    def __init__(self,data,imageidx,transform = None,data_aug = None):
        '''
        data = List of tuples containing input data for every modality. Last element of the tuple contains the label of the input.
        imageidx = index of the image modality within the input tuple. Equal to -1 if there is no image modality in the input
        transform = transform to perform on the image input before retreiving a sample from the dataset
        data_aug = data augmentation routine to perform on the image input before retreiving a sample from the dataset
        '''
        self.data= data
        self.imageidx = imageidx
        self.transform = transform
        self.data_aug = data_aug
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        
        sample = self.data[idx]
        if (self.transform is not None) and (self.imageidx != -1) :
            sample[self.imageidx] = self.transform(sample[self.imageidx])

        if (self.data_aug is not None) and (self.imageidx != -1):
            sample[self.imageidx] = self.data_aug(sample[self.imageidx])
            

        return sample


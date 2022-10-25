import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder



def get_data_loader(link, ds='train', create_encodings=False, batch_size=64, analysis=False):
    
    dataset = MicroRNADataset(link, ds, create_encodings)
    
    if ds == 'train' and not analysis:
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    elif ds == 'train' and analysis:
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    elif ds == 'test' and not analysis:
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
        
    
    return dataloader


class MicroRNADataset(Dataset):

    def __init__(self, link, ds='train', create_encodings=False, use_subset=False):
        
        # loading images
        self.link = link
        self.ds = ds
        self.images = np.load(f'{self.link}/data/modmirbase_{self.ds}_images.npz')['arr_0']/255
        
        
        if create_encodings:
            x_cat = self.get_encoded_values(self.images, self.ds)
        else:
            x_cat = np.load(f'{self.link}/data/modmirbase_{self.ds}_images_cattt.npz')['arr_0']
        #self.images_cat = np.load(f'{link}/data/modmirbase_{ds}_images_cat_new.npz')
        
        self.images_cat = x_cat
        
        mountain = np.load(f'{self.link}/data/modmirbase_{self.ds}_mountain.npy')
        mtop = np.flip(mountain[:,:100], 1)[:,None]
        mbot = np.abs(mountain[:,100:][:,None])
        self.mountain = np.concatenate([mtop, mbot], 1)
        # loading labels
        print('Loading Labels! (~10s)')     
        ohe = OneHotEncoder(categories='auto', sparse=False)
        labels = np.load(f'{self.link}/data/modmirbase_{self.ds}_labels.npz')['arr_0']
        self.labels = ohe.fit_transform(labels)
        
        
        #self.mountain = np.load(f'{link}/modmirbase_{ds}_mountain.npy')
        
        
        # loading names
        print('Loading Names! (~5s)')
        names =  np.load(f'{self.link}/data/modmirbase_{self.ds}_names.npz')['arr_0']
        names = [i.decode('utf-8') for i in names]
        self.species = ['mmu', 'prd', 'hsa', 'ptr', 'efu', 'cbn', 'gma', 'pma',
                        'cel', 'gga', 'ipu', 'ptc', 'mdo', 'cgr', 'bta', 'cin', 
                        'ppy', 'ssc', 'ath', 'cfa', 'osa', 'mtr', 'gra', 'mml',
                        'stu', 'bdi', 'rno', 'oan', 'dre', 'aca', 'eca', 'chi',
                        'bmo', 'ggo', 'aly', 'dps', 'mdm', 'ame', 'ppc', 'ssa',
                        'ppt', 'tca', 'dme', 'sbi']
        # assigning a species label to each observation from species
        # with more than 200 observations from past research
        self.names = []
        for i in names:
            append = False
            for j in self.species:
                if j in i.lower():
                    self.names.append(j)
                    append = True
                    break
            if not append:
                if 'random' in i.lower() or i.isdigit():
                    self.names.append('hsa')
                else:
                    self.names.append('notfound')
        
        # performing one hot encoding
        ohe = OneHotEncoder(categories='auto', sparse=False)
        
       
        
        self.names_ohe = ohe.fit_transform(np.array(self.names).reshape(-1,1))
            
    def __len__(self):
        return(self.images.shape[0])

    def __getitem__(self, idx):
        #d = self.names_ohe[idx]
        y = self.labels[idx]
        x = self.images[idx]
        x = np.transpose(x, (2,0,1))
        x_cat = self.images_cat[idx]
        m = self.mountain[idx]
        return (x_cat, x, y, m)


    def get_encoded_values(self, x, ds):
        """
        given an image or batch of images
        returns length of strand, length of bars and colors of bars
        """
        n = x.shape[0]
        x = np.transpose(x, (0,3,1,2))
        x_cat = np.zeros((n, 5, 25, 100), dtype=np.uint8)
        
        for i in range(n):
            if i % 100 == 0:
                print(f'at {i} out of {n}')
            for j in range(100):
                if (x[i,:,12,j] == np.array([1,1,1])).all():
                    break
                else:
                    # loop through all pixels of the bar
                    for k in range(25):
                        if (x[i,:,k,j] == np.array([1,1,1])).all():
                            continue
                        else:
                            x_cat[i,self.get_color(x[i,:,k,j]),k,j] = 1

        np.savez_compressed(f'{self.link}/data/modmirbase_{self.ds}_images_cattt.npz', x_cat)
        #with open(f'{link}/data/modmirbase_{ds}_images_cattt.npz', 'wb') as f:
        #    np.save(f, out_len)
        

        return x_cat

        
    
    def get_color(self, pixel):
        """
        returns the encoded value for a pixel
        """
        if (pixel == np.array([0,0,0])).all():  
            return 0 # black
        elif (pixel == np.array([1,0,0])).all():  
            return 1 # red
        elif (pixel == np.array([0,0,1])).all():  
            return 2 # blue
        elif (pixel == np.array([0,1,0])).all():  
            return 3 # green
        elif (pixel == np.array([1,1,0])).all():  
            return 4 # yellow
        else:
            print("Something wrong!")

import random
import sys
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


class Augmentor:
    def __init__(self, file_loc='D:/users/Marko/downloads/mirna/data/'):
        
        self.length_table = {('A','U'):2,('U','A'):2,('C','G'):1,
                    ('G','C'):1,('G','U'):3,('U','G'):3,
                    ('A','A'):4,('A','G'):4,('A','C'):4,
                    ('C','C'):4,('C','U'):4,('C','A'):4,
                    ('U','C'):4,('U','U'):4,('G','A'):4,
                    ('G','G'):1,('A','Z'):1,('Z','A'):1,
                    ('U','Z'):1,('Z','U'):1,('C','Z'):1,
                    ('Z','C'):1,('G','Z'):1,('Z','G'):1,
                    ('Z','Z'):1}
        self.images = np.load(f'{file_loc}/modmirbase_train_images.npz')['arr_0']/255
        
        x_len = np.load(f'{file_loc}/modmirbase_train_images_len.npz')
        x_bar = np.load(f'{file_loc}/modmirbase_train_images_bar.npz')
        self.x_len_max = np.argmax(x_len, 1)
        self.x_bar_max = np.argmax(x_bar, 1)
        
        self.parse_images()
        
    def parse_images(self):
        self.top = []
        self.bot = []
        self.top_bonds = []
        self.bot_bonds = []
        for i in tqdm(range(self.images.shape[0])):
            topstr=''
            botstr=''
            top_bond = []
            bot_bond = []
            for j in range(self.x_len_max[i]+1):
                top_col = self.get_color(self.images[i,12,j])
                bot_col = self.get_color(self.images[i,13,j])

                topstr+=top_col
                botstr+=bot_col

                if top_col == 'Z':
                    top_bond.append('-')
                elif self.x_bar_max[i, 2*j] == self.length_table[top_col, bot_col]:
                    top_bond.append('|')
                else:
                    top_bond.append('.')

                if bot_col == 'Z':
                    bot_bond.append('-')
                elif self.x_bar_max[i, 1+2*j] == self.length_table[top_col, bot_col]:
                    bot_bond.append('|')
                else:
                    bot_bond.append('.')
            self.top.append(topstr)
            self.bot.append(botstr)
            self.top_bonds.append(top_bond)
            self.bot_bonds.append(bot_bond)
    
    
    def augment_mirna(self, idx, swap_prob=.1, fill_prob=.2, remove_prob=.2,
                      strong_prob=.1, weak_prob=.1, mix_prob=.25,
                      reverse_prob=.2, chunk_size=9):
        lent = self.x_len_max[idx]
        tn = list(self.top[idx])
        bn = list(self.bot[idx])
        tb = self.top_bonds[idx].copy()
        bb = self.bot_bonds[idx].copy()
        
        tn,bn,tb,bb = self.swap_nucleotides(tn,bn,tb,bb,lent,swap_prob)
        tn,bn,tb,bb = self.change_gap(tn,bn,tb,bb,lent,fill_prob,remove_prob)
        tn,bn,tb,bb = self.change_bond(tn,bn,tb,bb,lent,strong_prob,weak_prob)
        tn,bn,tb,bb = self.mix_chunks(tn,bn,tb,bb,lent,mix_prob,reverse_prob,chunk_size)
        rec = self.reconstruct_image(tn,bn,tb,bb,lent)

        return rec
    
    def generate_augmented_mirna(self, out_folder, k):
        
        new_rna = np.ones((self.images.shape[0]*k,25,100,3), dtype=np.uint8)
        
        for i in tqdm(range(self.images.shape[0])):
            for j in range(k):
                new_rna[i*k + j] = self.augment_mirna(i)
        
        try:
            os.makedirs(out_folder)
        except:
            raise ValueError(f"Directory {out_folder} exists!")
        
        np.save(out_folder + '/new_mirna_data.npy', new_rna)
    
    def swap_nucleotides(self, top, bot, top_bond, bot_bond, lent, prob=.2):
        for i in range(8,lent):
            if np.random.uniform() < prob:
                top[i], bot[i] = bot[i],top[i]
                top_bond[i], bot_bond[i] = bot_bond[i], top_bond[i]

        return top, bot, top_bond, bot_bond

    def change_gap(self, top, bot, top_bond, bot_bond, lent, prob_fill=.2, prob_remove=.2):
        for i in range(8,lent):
            if top[i] != 'Z' and bot[i] != 'Z':
                if np.random.uniform() < prob_remove:
                    if np.random.uniform() < .5:
                        top[i] = "Z"
                        top_bond[i] = '-'
                        bot_bond[i] = '.'
                    else:
                        bot[i] = "Z"
                        top_bond[i] = '.'
                        bot_bond[i] = '-'

            else:
                if top[i] == "Z" and np.random.uniform() < prob_fill:
                    top[i] = self.get_matching_color(bot[i])
                    # we are getting a matching color -> always results in good bond
                    top_bond[i] = '|'
                    bot_bond[i] = '|'

                if bot[i] == "Z" and np.random.uniform() < prob_fill:
                    bot[i] = self.get_matching_color(top[i])
                    top_bond[i] = '|'
                    bot_bond[i] = '|'
        return top, bot, top_bond, bot_bond
    
    def change_bond(self, top, bot, top_bond, bot_bond, lent, prob_make_strong=.1, prob_make_weak=.1):
        for i in range(8, lent):
            if self.is_strong_bond(top[i],bot[i]):
                if np.random.uniform() < prob_make_weak:
                    top[i], bot[i] = self.get_weak_bond()
                    top_bond[i] = '.'
                    bot_bond[i] = '.'
            else:
                if np.random.uniform() < prob_make_strong:
                    top[i], bot[i] = self.get_strong_bond()
                    top_bond[i] = '|'
                    bot_bond[i] = '|'
        return top, bot, top_bond, bot_bond
    
    def mix_chunks(self, top, bot, top_bond, bot_bond, lent, prob=.3, prob_reverse=.3,chunk_length=9):
        chunks = []
        chunk_idx = np.arange((lent-8)//chunk_length+1)
        shuffle_list=[]
        shuffle_dict={}

        for i in chunk_idx:
            shuffle_dict[i] = i
            if np.random.uniform()<prob_reverse:
                chunks.append(np.arange(8+i*chunk_length, min(8+(i+1)*chunk_length, lent+1))[::-1])
            else:
                chunks.append(np.arange(8+i*chunk_length, min(8+(i+1)*chunk_length, lent+1)))
            if np.random.uniform()<prob:
                shuffle_list.append(i)
        shuf = shuffle_list.copy()
        random.shuffle(shuf)

        for i in range(len(shuffle_list)):
            shuffle_dict[shuffle_list[i]] = shuf[i] 

        chunks = [chunks[i] for i in shuffle_dict.values()]

        order = [i for i in range(8)]+[i for item in chunks for i in item]
        top = [top[i] for i in order]
        bot = [bot[i] for i in order]

        top_bond = [top_bond[i] for i in order]
        bot_bond = [bot_bond[i] for i in order]
        return top, bot, top_bond, bot_bond

    def get_color(self, pixel):
        """
        returns the nucleotide for a pixel
        """
        if (pixel == np.array([0,0,0])).all():  
            return 'Z' # black
        elif (pixel == np.array([1,0,0])).all():  
            return 'G' # red
        elif (pixel == np.array([0,0,1])).all():  
            return 'C' # blue
        elif (pixel == np.array([0,1,0])).all():  
            return 'U' # green
        elif (pixel == np.array([1,1,0])).all():  
            return 'A' # yellow
        else:
            print("Something wrong!")
        
    def get_pixel(self, color):
        """
        returns the pixel for a nucleotide
        """
        if color == 'Z': 
            return np.array([0,0,0])
        elif color == 'G':
            return np.array([1,0,0])
        elif color == 'C':
            return np.array([0,0,1])
        elif color == 'U': 
            return np.array([0,1,0])
        elif color == 'A':
            return np.array([1,1,0])
        else:
            print("Something wrong!")
            
    def get_matching_color(self, nt):
        if nt == 'A':
            return 'U'
        if nt == 'U':
            return np.random.choice(['A','G'], p=[.7,.3])
        if nt == 'C':
            return 'G'
        if nt == 'G':
            return np.random.choice(['C','U'], p=[.8,.2])
        if nt == 'Z':
            return 'Z'
        
    def is_strong_bond(self, nt1, nt2):
        if (nt1 == 'A' and nt2 == 'U') or (nt1 == 'U' and nt2 == 'A'):
            return True
        elif (nt1 == 'C' and nt2 == 'G') or (nt1 == 'G' and nt2 == 'C'):
            return True
        elif (nt1 == 'U' and nt2 == 'G') or (nt1 == 'G' and nt2 == 'U'):
            return True
        else:
            return False 
        
    def get_weak_bond(self):
        choices = [('A','A'),('A','G'),('A','C'),('C','C'),('C','U'),('C','A'),('U','C'),('U','U'),('G','A'),('G','G')]
        return choices[np.random.choice(len(choices))]
    
    def get_strong_bond(self):
        choices = [('A','U'),('U','A'),('C','G'),('G','C'),('G','U'),('U','G')]
        return choices[np.random.choice(len(choices))]
    
    def reconstruct_image(self, top, bot, top_bond, bot_bond, rna_length):

        image = np.ones((25,100,3))
        
        for i in range(len(top)):
            lnt = self.length_table[(top[i],bot[i])]

            lnt_top = self.calc_length(top, bot, top_bond, bot_bond, lnt, i, 'top', rna_length)
            lnt_bot = self.calc_length(top, bot, top_bond, bot_bond, lnt, i, 'bot', rna_length)


            image[max(12-lnt_top,0):13,i] = self.get_pixel(top[i])
            image[13:min(13+lnt_bot+1,25),i] = self.get_pixel(bot[i])
        return image

    def calc_length(self, top, bot, top_bond, bot_bond, lnt, pos, strand, rna_length):
        if strand == 'top':
            strand = top_bond
        else:
            strand = bot_bond

        if strand[pos] == '.':
            tr = pos - 1
            tl = pos + 1



            while tr >= 0 and strand[tr] == '.':
                tr -= 1
            while tl <= rna_length and strand[tl] =='.':
                tl += 1

            dots = tr+tl+1
            mid = dots/2
            p = pos - tr

            if p > mid:
                k = dots+1
                while k > p:
                    lnt += 2
                    k -= 1
            else:
                k = 1
                while k <= p:
                    lnt += 2
                    k += 1
        return lnt
    
    
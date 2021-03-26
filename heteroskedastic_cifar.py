import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import copy
import warnings
from numpy.testing import assert_array_almost_equal

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]
    y = np.array(y)
    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y.tolist()

class HETEROSKEDASTICCIFAR10(torchvision.datasets.CIFAR10):
    
    def __init__(self, root, mislabel_type='hetero', mislabel_ratio=0.5, imb_type=None, imb_ratio=0.1, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(HETEROSKEDASTICCIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.num_cls = np.max(self.targets) + 1
        np.random.seed(rand_number)
        self.gen_mislabeled_data(mislabel_type=mislabel_type, mislabel_ratio=mislabel_ratio)
        if self.num_cls == 10 and imb_type == 'step':    
            self.cls_num_list = [5000, int(5000*imb_ratio), 5000, int(5000*imb_ratio), 5000, int(5000*imb_ratio), 5000, 5000, 5000, int(5000*imb_ratio)]
            self.gen_imbalanced_data(self.cls_num_list)
        
    def gen_imbalanced_data(self, img_num_per_cls):
        """Gen a list of imbalanced training data, and replace the origin data with the generated ones."""
        new_data = []
        new_targets = []
        new_real_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        real_targets_np = np.array(self.real_targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * len(selec_idx))
            new_real_targets.append(real_targets_np[selec_idx])
        new_data = np.vstack(new_data)
        new_real_targets = np.hstack(new_real_targets)
        self.data = new_data
        self.targets = new_targets
        self.real_targets = new_real_targets.tolist()

    def gen_mislabeled_data(self, mislabel_type, mislabel_ratio):
        """Gen a list of imbalanced training data, and replace the origin data with the generated ones."""
        new_targets = []
        
        if mislabel_type == 'agnostic':
            for i, target in enumerate(self.targets):
                if np.random.rand() < mislabel_ratio:
                    new_target = target
                    while new_target == target:
                        new_target = np.random.randint(self.num_cls)
                    new_targets.append(new_target)
                else:
                    new_targets.append(target)
        elif mislabel_type == 'hetero':
            if self.num_cls == 10:
                P = np.eye(10)
                manip_list1 = [1, 9]
                manip_list2 = [3, 5]
                for idx in manip_list1:
                    P[idx, idx] -= 0.8
                    P[idx, manip_list1] += 0.4
                for idx in manip_list2:
                    P[idx, idx] -= 0.8
                    P[idx, manip_list2] += 0.4
            else:
                P = np.eye(self.num_cls)
                manip_list1 = [8,13,58,90,48]
                manip_list2 = [41,69,81,85,89]
                for idx in manip_list1:
                    P[idx, idx] -= 0.5
                    P[idx, manip_list1] += 0.1
                for idx in manip_list2:
                    P[idx, idx] -= 0.5
                    P[idx, manip_list2] += 0.1
            new_targets = multiclass_noisify(self.targets, P)
        else:
            warnings.warn('Noise type is not listed')

        self.real_targets = self.targets
        self.targets = new_targets

    def estimate_label_acc(self):
        targets_np = np.array(self.targets) 
        real_targets_np = np.array(self.real_targets)
        label_acc = np.sum((targets_np == real_targets_np)) / len(targets_np)
        return label_acc

    def __getitem__(self, index):  
        img, target, real_target = self.data[index], self.targets[index], self.real_targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, real_target, index   

class HETEROSKEDASTICCIFAR100(HETEROSKEDASTICCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    
if __name__ == '__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = HETEROSKEDASTICCIFAR10(root='./data', train=True,
                    download=True, transform=transform, imb_type='step')
    trainloader = iter(trainset)
    data, label, _, ind = next(trainloader)
    import pdb; pdb.set_trace()
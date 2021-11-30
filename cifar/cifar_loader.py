

class DataReader(Dataset):
    def __init__(self, input_path, size=150, is_train=False):

        self.size = size

        self.classes = ['AP2', 'AP3', 'AP4', 'AP5', 'PLAX', 'PSAX-AP', 'PSAX-AV', 'PSAX-MID',

                 'PSAX-MV', 'NONE']

        self.input_path = input_path

        

        self.reverse_label = dict()

        for i, x in enumerate(self.classes):

            self.reverse_label[x] = i

        

        self.sampes = []

        for x in self.classes:

            files = os.listdir(self.input_path + "/" + x)

            for file in files:

                self.sampes.append((self.input_path + "/" + x + "/" + file,

                                    self.reverse_label[x]))

        np.random.shuffle(self.sampes)

        if is_train:

            self.transform =  transforms.Compose([

                                        transforms.RandomResizedCrop(size=(size, size), scale=(0.75, 1.0)),

                                        transforms.RandomRotation(degrees=20, fill=0),

                                        transforms.RandomHorizontalFlip(p=0.5),

                                        transforms.RandomVerticalFlip(p=0.5),

                                        transforms.ToTensor()

                                    ])

        else:

            self.transform =  transforms.Compose([

                                        transforms.Resize(size=(size, size)),

                                        transforms.ToTensor()

                                    ])

            

    def __len__(self):

        return len(self.sampes)

    

    def __getitem__(self, idx):

        file, label = self.sampes[idx]

        image = Image.open(file)

        ret = self.transform(image)

        

        label = torch.tensor(label, dtype=torch.int64)

        return ret, label
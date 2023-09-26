

transform = T.Compose(
            [
                # T.ToTensor(),
                T.RandomHorizontalFlip(),
                T.Resize(size=(224, 224))
            ]
        )

# preprocess = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )])

class SnacksTriplets(Dataset):
    def __init__(self, df=None, transform=None, num_triplets=10, classes=None,
                 train=False):

        self.df = df

        self.train = train

        self.transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

        self.num_triplets = num_triplets
        self.classes = classes

        self.take_df = self.df.sample(n=self.num_triplets)


    def __len__(self):
        return len(self.take_df)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # anchor is idx
        anchor = self.df.iloc[idx, 1]
        anchor_label = self.df.iloc[idx, 0]

        # get positive

        # print('idx:', idx)

        # if self.train:
        positive = anchor
        while positive == anchor:
            positive = self.df[self.df[0]==anchor_label][1].sample(n=1).values[0]
        # else:
        #     positive = self.df.iloc[self.df.index!=idx, 1]
        #     positive = positive[positive[0]==anchor_label][1][0]

        # print('positive:', positive)

        # get negative

        # if self.train:
        negative = self.df[self.df[0]!=anchor_label][1].sample(n=1).values[0]
        # else:
        #     negative = self.df[self.df[0]!=anchor_label][1][0]

        # print('negative:', negative)


        # get anchor image
        anchor_img = Image.open(anchor)
        # transform
        anchor_img = np.array(self.transform(anchor_img)).reshape(3, 224, 224)

        # get positive image
        positive_img = Image.open(positive)
        # transform
        positive_img = np.array(self.transform(positive_img)).reshape(3, 224, 224)

        # get negative image
        negative_img = Image.open(negative)
        # transform
        negative_img = np.array(self.transform(negative_img)).reshape(3, 224, 224)

        return anchor_img, positive_img, negative_img




NUM_WORKERS = 0 if str(device)=='cpu' else 2



def create_triplet_loader(df, batch_size:int, num_triplets:int, classes:set, transform, train):
    # print('ok1')
    ds = SnacksTriplets(df=df,
                        transform = transform,
                        num_triplets=num_triplets,
                        classes=classes,
                        train=train)
    # print('ok2'); exit(0)
    return DataLoader(
      ds,
      batch_size=batch_size,
      #num_workers=16,
      num_workers=NUM_WORKERS,
      shuffle=True
    )


BATCH_SIZE = 32
# num_triplets = 50
num_triplets = df_train.shape[0]

train_triplet_loader = create_triplet_loader(df_train,
                                             BATCH_SIZE,
                                             num_triplets,
                                             list(classes),
                                             transform,
                                             True
                                             )
num_triplets = df_val.shape[0]
val_triplet_loader = create_triplet_loader(df_val,
                                           BATCH_SIZE,
                                           num_triplets,
                                           list(classes),
                                           transform,
                                           False
                                           )
num_triplets = df_test.shape[0]
test_triplet_loader = create_triplet_loader(df_test,
                                           BATCH_SIZE,
                                           num_triplets,
                                           list(classes),
                                           transform,
                                           False
                                           )

print('df_train:', df_train.shape,
      'df_val:', df_val.shape,
      'df_test:', df_test.shape
      )

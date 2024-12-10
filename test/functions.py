import torch
import torch.utils.data as data


# for changing size of AEV, change trainingpath, change input size (840 or 560), change SummaryWriter
class NMR(torch.nn.Module):
    def __init__(self, embedding=0):
        super().__init__()
        self.layer0 = torch.nn.Embedding(
            20, embedding, max_norm=2
        )  # Num=# of residues, size=variable, max_norm=None (default)
        self.layer1 = torch.nn.Linear(560 + embedding, 256)  # 560 if 1.0, 840 if 1.5
        self.layer2 = torch.nn.Linear(256, 64)
        self.layer3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        indices = x[:, -1].long()
        x = x[:, :-1]
#        print("indices", indices)
        embedded_vector = self.layer0(indices)
        x = torch.cat([x, embedded_vector], 1)
        y = self.layer1(x)
        y = torch.nn.functional.elu(y)
        y = self.layer2(y)
        y = torch.nn.functional.elu(y)
        y = self.layer3(y)
        return y.flatten()


class ShiftsDataset(data.Dataset):
    def __init__(self, trainingset):
        self.trainingset = trainingset

    def __getitem__(self, i):
        one_aev = self.trainingset[0][i]
        one_shift = self.trainingset[1][i]
        one_res_idx = self.trainingset[2][i]
        return one_aev, one_shift, one_res_idx

    def __len__(self):
        one_aev_size = len(self.trainingset[0])
        return one_aev_size


# dict of means and standard deviations for each atom type according to residue type derived from the training set.
# Ensembling (Means from entire trainingset)
ens_means = {
    #    1H-N    #
    "H": [
        8.198,
        8.226,
        8.384,
        8.315,
        8.356,
        8.330,
        8.195,
        8.318,
        8.257,
        8.348,
        8.226,
        8.198,
        8.241,
        8.343,
        0.000,
        8.301,
        8.305,
        8.299,
        8.311,
        8.317,
    ],
    #    1HA    #
    "HA": [
        4.337,
        4.337,
        4.725,
        4.641,
        4.769,
        4.277,
        4.325,
        4.175,
        4.559,
        4.303,
        4.390,
        4.329,
        4.418,
        4.664,
        4.382,
        4.585,
        4.575,
        4.794,
        4.713,
        4.275,
    ],
    #    13CA    #
    "CA": [
        53.364,
        57.021,
        53.700,
        54.764,
        57.437,
        57.765,
        56.709,
        45.650,
        56.828,
        61.810,
        55.666,
        57.093,
        56.425,
        58.440,
        63.583,
        58.767,
        62.255,
        58.016,
        58.083,
        62.549,
    ],
    #    13CB    #
    "CB": [
        19.413,
        30.799,
        38.923,
        41.193,
        35.288,
        30.237,
        29.399,
        0.000,
        30.407,
        38.918,
        42.469,
        33.077,
        33.644,
        40.162,
        31.920,
        64.319,
        70.078,
        30.350,
        39.738,
        32.968,
    ],
    #    13C=O    #
    "C": [
        178.012,
        176.791,
        175.452,
        176.702,
        174.248,
        177.388,
        176.580,
        173.912,
        175.200,
        175.935,
        177.158,
        176.862,
        176.292,
        175.719,
        176.904,
        174.529,
        174.456,
        176.404,
        175.484,
        175.808,
    ],
    #    15N    #
    "N": [
        122.895,
        120.190,
        119.127,
        120.353,
        119.184,
        120.104,
        119.534,
        108.945,
        118.897,
        121.539,
        121.749,
        120.772,
        119.761,
        120.169,
        133.858,
        115.916,
        115.101,
        121.510,
        120.404,
        121.005,
    ],
}

ens_stdevs = {
    #    1H-N    #
    "H": [
        0.719,
        0.678,
        0.710,
        0.625,
        0.825,
        0.695,
        0.688,
        0.861,
        0.863,
        0.802,
        0.744,
        0.676,
        0.718,
        0.787,
        0.000,
        0.726,
        0.724,
        0.840,
        0.839,
        0.769,
    ],
    #    1HA    #
    "HA": [
        0.520,
        0.514,
        0.462,
        0.364,
        0.727,
        0.474,
        0.529,
        0.411,
        0.576,
        0.638,
        0.550,
        0.500,
        0.541,
        0.654,
        0.401,
        0.499,
        0.545,
        0.626,
        0.655,
        0.585,
    ],
    #    13CA    #
    "CA": [
        2.013,
        2.361,
        1.854,
        2.087,
        3.032,
        2.147,
        2.178,
        1.244,
        2.564,
        2.663,
        2.082,
        2.175,
        2.191,
        2.691,
        1.413,
        2.203,
        2.622,
        2.388,
        2.652,
        2.860,
    ],
    #    13CB    #
    "CB": [
        1.957,
        1.883,
        1.785,
        1.590,
        6.921,
        1.838,
        1.982,
        0.000,
        2.484,
        1.969,
        1.917,
        1.811,
        2.407,
        2.133,
        1.197,
        1.700,
        1.568,
        2.135,
        2.205,
        1.697,
    ],
    #    13C=O    #
    "C": [
        2.110,
        2.041,
        1.808,
        1.715,
        1.657,
        1.991,
        1.965,
        1.701,
        2.130,
        1.870,
        1.938,
        1.883,
        2.154,
        1.986,
        1.624,
        1.749,
        1.644,
        2.145,
        1.905,
        1.810,
    ],
    #    15N    #
    "N": [
        3.890,
        4.140,
        4.331,
        4.295,
        4.769,
        3.634,
        3.999,
        3.915,
        4.581,
        4.633,
        4.177,
        3.955,
        4.060,
        4.461,
        8.191,
        4.182,
        5.305,
        4.788,
        4.704,
        5.134,
    ],
}

ens_means = {k: torch.tensor(v, dtype=torch.float32) for k, v in ens_means.items()}
ens_stdevs = {k: torch.tensor(v, dtype=torch.float32) for k, v in ens_stdevs.items()}

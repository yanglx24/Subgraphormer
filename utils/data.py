import utils.preprocess as u_prep
import torch_geometric.data as data
import torch_geometric as pyg
import torch
import shutil
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.datasets import GNNBenchmarkDataset, LRGBDataset
import pandas as pd
import hashlib
import os.path as osp
from tqdm import tqdm
import pickle
from torch_geometric.data import Data

try:
    # Code you want to try to run
    from ogb.utils.mol import smiles2graph
except Exception as e:
    # Code to run in case of a ZeroDivisionError
    print("An error occurred:", e)
    print("Can't import  smile2graph!")
    smiles2graph = 1


def get_dataloader(cfg, sample_idx=0):
    transform_suffix = ""
    if "Peptides" in cfg.data.name:
        transform_suffix = (
            "_sample_keep_prob_"
            + str(float(cfg.data.sampling.keep_subgraph_prob))
            + "_sample_index_"
            + str(sample_idx)
        )
    transforms = u_prep.subgraph_construction(cfg)
    if cfg.data.name in [
        "PascalVOC-SP",
        "COCO-SP",
        "PCQM-Contact",
        "Peptides-func",
        "Peptides-struct",
    ]:
        dataloader, num_elements_in_target = get_LRGB_dataloader(
            cfg=cfg, transform_suffix=transform_suffix, transforms=transforms
        )
    elif cfg.data.name in ["MNIST", "CIFAR10", "PATTERN", "CLUSTER"]:
        dataloader, num_elements_in_target = get_gnnbenchmark_dataloader(
            cfg=cfg,
            transform_suffix=transform_suffix,
            transforms=transforms,
        )
    elif cfg.data.name in ["zinc12k", "zinc-full"]:
        dataloader, num_elements_in_target = get_zinc_dataloader(
            cfg=cfg, transform_suffix=transform_suffix, transforms=transforms
        )
    else:
        raise ValueError(f"No dataset available for: {cfg.data.name}")
    assert (
        num_elements_in_target == cfg.model.final_dim
    ), f"The final dim: {cfg.model.final_dim} should match the number of target elements: {num_elements_in_target}!"
    return dataloader


# ----------------------------- dataloaders ----------------------------- #


def get_LRGB_dataloader(cfg, transform_suffix, transforms):
    if cfg.data.name == "Peptides-func":
        num_elements_in_target = 10
    elif cfg.data.name == "Peptides-struct":
        num_elements_in_target = 11
    dataloader = {
        dataset_type: data.DataLoader(
            LRGBDataset(
                name=cfg.data.name,
                split=dataset_type,
                transform=transforms,
                root=cfg.data.dir + transform_suffix,
            )
        )
        for dataset_type in ["train", "val", "test"]
    }
    return dataloader, num_elements_in_target


def get_zinc_dataloader(cfg, transform_suffix, transforms):
    is_subset = "12" in cfg.data.name
    dataloader = {
        name: data.DataLoader(
            pyg.datasets.ZINC(
                split=name,
                subset=is_subset,
                root=cfg.data.dir + transform_suffix,
                transform=transforms,
            ),
            batch_size=cfg.data.bs,
            num_workers=cfg.data.num_workers,
            shuffle=(name == "train"),
        )
        for name in ["train", "val", "test"]
    }
    num_elements_in_target = 1
    return dataloader, num_elements_in_target


def get_gnnbenchmark_dataloader(cfg, transform_suffix, transforms):
    if cfg.data.name in ["MNIST", "CIFAR10"]:
        num_elements_in_target = 10
    elif cfg.data.name == "PATTERN":
        num_elements_in_target = 2
    elif cfg.data.name == "CLUSTER":
        num_elements_in_target = 6
    dataloader = {
        name: data.DataLoader(
            GNNBenchmarkDataset(
                root=cfg.data.dir + transform_suffix,
                name=cfg.data.name,
                split=name,
                transform=transforms,
            ),
            batch_size=cfg.data.bs,
            num_workers=cfg.data.num_workers,
            shuffle=(name == "train"),
        )
        for name in ["train", "val", "test"]
    }
    return dataloader, num_elements_in_target


def get_peptides_func_dataloader(cfg, transform_suffix, transforms):
    dataset = PeptidesFunctionalDataset(
        root=cfg.data.dir + "_using_pre_transform_" + transform_suffix,
        pre_transform=transforms,
    )
    splits = dataset.get_idx_split()
    train_dataset = dataset[splits["train"]]
    val_dataset = dataset[splits["val"]]
    test_dataset = dataset[splits["test"]]
    dataloader = {
        "train": data.DataLoader(
            train_dataset,
            batch_size=cfg.data.bs,
            num_workers=cfg.data.num_workers,
            shuffle=True,
        ),
        "val": data.DataLoader(
            val_dataset,
            batch_size=cfg.data.bs,
            num_workers=cfg.data.num_workers,
            shuffle=False,
        ),
        "test": data.DataLoader(
            test_dataset,
            batch_size=cfg.data.bs,
            num_workers=cfg.data.num_workers,
            shuffle=False,
        ),
    }
    num_elements_in_target = 10
    return dataloader, num_elements_in_target


def get_peptides_struct_dataloader(cfg, transform_suffix, transforms):
    dataset = PeptidesStructuralDataset(
        root=cfg.data.dir + "_using_pre_transform_" + transform_suffix,
        pre_transform=transforms,
    )
    splits = dataset.get_idx_split()
    train_dataset = dataset[splits["train"]]
    val_dataset = dataset[splits["val"]]
    test_dataset = dataset[splits["test"]]
    dataloader = {
        "train": data.DataLoader(
            train_dataset,
            batch_size=cfg.data.bs,
            num_workers=cfg.data.num_workers,
            shuffle=True,
        ),
        "val": data.DataLoader(
            val_dataset,
            batch_size=cfg.data.bs,
            num_workers=cfg.data.num_workers,
            shuffle=False,
        ),
        "test": data.DataLoader(
            test_dataset,
            batch_size=cfg.data.bs,
            num_workers=cfg.data.num_workers,
            shuffle=False,
        ),
    }
    num_elements_in_target = 11
    return dataloader, num_elements_in_target


# ----------------------------- datasets ----------------------------- #


class PeptidesFunctionalDataset(InMemoryDataset):
    def __init__(
        self,
        root="datasets",
        smiles2graph=smiles2graph,
        transform=None,
        pre_transform=None,
    ):
        """
        PyG dataset of 15,535 peptides represented as their molecular graph
        (SMILES) with 10-way multi-task binary classification of their
        functional classes.

        The goal is use the molecular representation of peptides instead
        of amino acid sequence representation ('peptide_seq' field in the file,
        provided for possible baseline benchmarking but not used here) to test
        GNNs' representation capability.

        The 10 classes represent the following functional classes (in order):
            ['antifungal', 'cell_cell_communication', 'anticancer',
            'drug_delivery_vehicle', 'antimicrobial', 'antiviral',
            'antihypertensive', 'antibacterial', 'antiparasitic', 'toxic']

        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "peptides-functional")

        self.url = "https://www.dropbox.com/s/ol2v01usvaxbsr8/peptide_multi_class_dataset.csv.gz?dl=1"
        self.version = (
            "701eb743e899f4d793f0e13c8fa5a1b4"  # MD5 hash of the intended dataset file
        )
        self.url_stratified_split = "https://www.dropbox.com/s/j4zcnx2eipuo0xz/splits_random_stratified_peptide.pickle?dl=1"
        self.md5sum_stratified_split = "5a0114bdadc80b94fc7ae974f13ef061"

        # Check version and update if necessary.
        release_tag = osp.join(self.folder, self.version)
        if osp.isdir(self.folder) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.folder)

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "peptide_multi_class_dataset.csv.gz"

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.raw_dir)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(osp.join(self.root, hash), "w").close()
            # Download train/val/test splits.
            path_split1 = download_url(self.url_stratified_split, self.root)
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "peptide_multi_class_dataset.csv.gz")
        )
        smiles_list = data_df["smiles"]

        print("Converting SMILES strings into graphs...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            graph = self.smiles2graph(smiles)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.y = torch.Tensor([eval(data_df["labels"].iloc[i])])

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]
            # data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        """Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = osp.join(self.root, "splits_random_stratified_peptide.pickle")
        with open(split_file, "rb") as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        return split_dict


class PeptidesStructuralDataset(InMemoryDataset):
    def __init__(
        self,
        root="datasets",
        smiles2graph=smiles2graph,
        transform=None,
        pre_transform=None,
    ):
        """
        PyG dataset of 15,535 small peptides represented as their molecular
        graph (SMILES) with 11 regression targets derived from the peptide's
        3D structure.

        The original amino acid sequence representation is provided in
        'peptide_seq' and the distance between atoms in 'self_dist_matrix' field
        of the dataset file, but not used here as any part of the input.

        The 11 regression targets were precomputed from molecule XYZ:
            Inertia_mass_[a-c]: The principal component of the inertia of the
                mass, with some normalizations. Sorted
            Inertia_valence_[a-c]: The principal component of the inertia of the
                Hydrogen atoms. This is basically a measure of the 3D
                distribution of hydrogens. Sorted
            length_[a-c]: The length around the 3 main geometric axis of
                the 3D objects (without considering atom types). Sorted
            Spherocity: SpherocityIndex descriptor computed by
                rdkit.Chem.rdMolDescriptors.CalcSpherocityIndex
            Plane_best_fit: Plane of best fit (PBF) descriptor computed by
                rdkit.Chem.rdMolDescriptors.CalcPBF
        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "peptides-structural")

        ## Unnormalized targets.
        # self.url = 'https://www.dropbox.com/s/464u3303eu2u4zp/peptide_structure_dataset.csv.gz?dl=1'
        # self.version = '9786061a34298a0684150f2e4ff13f47'

        ## Standardized targets to zero mean and unit variance.
        self.url = "https://www.dropbox.com/s/0d4aalmq4b4e2nh/peptide_structure_normalized_dataset.csv.gz?dl=1"
        self.version = (
            "c240c1c15466b5c907c63e180fa8aa89"  # MD5 hash of the intended dataset file
        )

        self.url_stratified_split = "https://www.dropbox.com/s/9dfifzft1hqgow6/splits_random_stratified_peptide_structure.pickle?dl=1"
        self.md5sum_stratified_split = "5a0114bdadc80b94fc7ae974f13ef061"

        # Check version and update if necessary.
        release_tag = osp.join(self.folder, self.version)
        if osp.isdir(self.folder) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.folder)

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "peptide_structure_normalized_dataset.csv.gz"

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.raw_dir)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(osp.join(self.root, hash), "w").close()
            # Download train/val/test splits.
            path_split1 = download_url(self.url_stratified_split, self.root)
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "peptide_structure_normalized_dataset.csv.gz")
        )
        smiles_list = data_df["smiles"]
        target_names = [
            "Inertia_mass_a",
            "Inertia_mass_b",
            "Inertia_mass_c",
            "Inertia_valence_a",
            "Inertia_valence_b",
            "Inertia_valence_c",
            "length_a",
            "length_b",
            "length_c",
            "Spherocity",
            "Plane_best_fit",
        ]
        # Assert zero mean and unit standard deviation.
        assert all(abs(data_df.loc[:, target_names].mean(axis=0)) < 1e-10)
        assert all(abs(data_df.loc[:, target_names].std(axis=0) - 1.0) < 1e-10)

        print("Converting SMILES strings into graphs...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][target_names]
            graph = self.smiles2graph(smiles)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.y = torch.Tensor([y])

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]
            # data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        """Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = osp.join(
            self.root, "splits_random_stratified_peptide_structure.pickle"
        )
        with open(split_file, "rb") as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        return split_dict

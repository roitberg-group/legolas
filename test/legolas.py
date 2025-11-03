import os
import collections
import re
import math
import argparse
from pathlib import Path
from typing import Optional

import torch
import pandas as pd
import numpy as np
import mdtraj as md
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure

from torchani.utils import ChemicalSymbolsToInts
from torchani.aev import AEVComputer, ANIAngular, ANIRadial
from functions import NMR, ens_means, ens_stdevs

NUM_MODELS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EntryPDB:
    """
    Class to hold the molecular data for one PDB.

    Attributes:
        species: list of atomic species
        coordinates: torch tensor of atomic coordinates
        res_idx: torch tensor of residue indices
        indices: a dictionary holding the indices for each atom type
    """

    SUPPORTED_SPECIES = ["H", "C", "N", "O", "S"]
    ALLOWED_HETFLAG = ["", "W", "H_DOD"]

    def __init__(self, structure: Structure):
        self.atypes = []
        self.species = []
        self.chemical_shifts = []
        self.coordinates = []
        self.res_idx = []
        self.seq_id = []
        self.unsupported = False
        self.indices = collections.defaultdict(lambda: [])
        self.str2int = ChemicalSymbolsToInts(self.SUPPORTED_SPECIES)
        # resname_dict = self.get_resname_dict()
        resname_dict, id_to_resname = self.get_resname_dict()
        self.id_to_resname = id_to_resname

        i = 0
        for atom in structure.get_atoms():
            hetflag, seqid, _ = atom.get_parent().get_id()
            if hetflag.strip() not in self.ALLOWED_HETFLAG:
                continue

            self.seq_id.append(seqid)

            atype = re.sub(r"\d", "", atom.get_name())
            self.atypes.append(atype)
            if atype == "H" and atom.get_parent().get_resname() == "HOH":
                continue
            if atype in interested_atypes:
                self.indices[atype].append(i)

            self.res_idx.append(resname_dict[atom.get_parent().get_resname()])

            if hasattr(atom, "cs"):
                cs = atom.cs
            else:
                cs = math.nan

            self.chemical_shifts.append(cs)
            self.coordinates.append(atom.get_coord())

            element = self.convert_element(atom.element)
            if element is None:
                self.unsupported = True
                self.unsupported_atype = atype
            self.species.append(element)
            i += 1

        self.chemical_shifts = torch.tensor(self.chemical_shifts)
        self.coordinates = torch.tensor(
            np.array(self.coordinates), dtype=torch.float32
        ).unsqueeze(0)
        self.res_idx = torch.tensor(self.res_idx, dtype=torch.long)
        self.species = self.str2int(self.species)
        for atype in self.indices:
            self.indices[atype] = torch.tensor(self.indices[atype], dtype=torch.long)

    @staticmethod
    def convert_element(element):
        if element in EntryPDB.SUPPORTED_SPECIES:
            return element
        if element == "D":
            return "H"
        raise ValueError("Bad element type")

    @staticmethod
    def get_resname_dict():
        string = """ala arg asn asp cys glu gln gly his ile leu lys met phe pro ser thr trp tyr val hoh dod""".split()
        otherlist = [item.upper() for item in string]
        name_to_id = {name: j for j, name in enumerate(otherlist)}
        id_to_name = {j: name for name, j in name_to_id.items()}
        return name_to_id, id_to_name
        # return {name: j for j, name in enumerate(otherlist)}

    def to(self, device):
        """Move tensors to the specified device"""
        self.seq_id = torch.tensor(self.seq_id).to(device)
        self.chemical_shifts = self.chemical_shifts.to(device)
        self.coordinates = self.coordinates.to(device)
        self.res_idx = self.res_idx.to(device)
        self.species = self.species.to(device)
        for atype in self.indices:
            self.indices[atype] = self.indices[atype].to(device)
        return self


class EntryMdtraj(EntryPDB):
    """
    Subclass of Entry specifically for molecular dynamics trajectories.

    Attributes:
        same as Entry, but the coordinates are a trajectory of all frames
    """

    def __init__(self, trajectory: md.Trajectory):
        self.atypes = []
        self.species = []
        self.chemical_shifts = []
        self.coordinates = []
        self.res_idx = []
        self.seq_id = []
        self.unsupported = False
        self.indices = collections.defaultdict(lambda: [])
        self.str2int = ChemicalSymbolsToInts(self.SUPPORTED_SPECIES)
        #        resname_dict = self.get_resname_dict()
        resname_dict, id_to_resname = self.get_resname_dict()
        self.id_to_resname = id_to_resname

        for i, atom in enumerate(trajectory.topology.atoms):
            element = self.convert_element(atom.element.symbol)

            self.seq_id = [atom.residue.resSeq for atom in trajectory.topology.atoms]

            atype = re.sub(r"\d", "", atom.name)
            self.atypes.append(atype)
            # Biopython and mdtraj should both ignore H if H has resname of “WAT” or “HOH”, and should only ignore H but keep O.
            # This is because for the trained model, the aev can only see the water's O atom, and H got ignored
            if (
                atype == "H" and atom.residue.name == "HOH"
            ):  # or atom.residue.name == "HOH":
                continue
            if atype in interested_atypes:
                self.indices[atype].append(i)

            self.res_idx.append(resname_dict[atom.residue.name])

            # cs not handeled, seems useless...
            if element is None:
                self.unsupported = True
                self.unsupported_atype = atype

            self.species.append(element)
            self.coordinates.append(trajectory.xyz[:, i, :])

        # Convert the species and coordinates lists to tensors
        self.seq_id = torch.tensor(self.seq_id, dtype=torch.long)
        self.chemical_shifts = torch.tensor(self.chemical_shifts)
        self.coordinates = torch.tensor(self.coordinates, dtype=torch.float32)
        self.coordinates = torch.transpose(
            self.coordinates, 0, 1
        ).contiguous()  # fix formatting issue
        #        self.coordinates = torch.tensor(trajectory.xyz, dtype=torch.float32)
        self.coordinates = self.coordinates * 10  # convert to angstrom
        self.res_idx = torch.tensor(self.res_idx, dtype=torch.long)
        self.species = self.str2int(self.species)
        #        print("coords", self.coordinates.shape)
        #        print("species", self.species.shape)
        for atype in self.indices:
            self.indices[atype] = torch.tensor(self.indices[atype], dtype=torch.long)


class ChemicalShiftPredictor(torch.nn.Module):
    """
    Class to predict the chemical shift of the atoms in a molecular simulation.

    Attributes:
        models: a ModuleDict of models for each atom type
        aev_computer: computes atomic environment vectors (AEVs)
        ens_means: ensemble mean chemical shifts
        ens_stdevs: ensemble standard deviations of chemical shifts
    """

    def __init__(self, model_paths):
        super(ChemicalShiftPredictor, self).__init__()

        # Load all models for different atom types
        self.models = self._load_models(model_paths)

        radial_terms = ANIRadial.like_2x()
        angular_terms = ANIAngular.like_2x()
        self.aev_computer = AEVComputer(
            angular=angular_terms,
            radial=radial_terms,
            num_species=5,
            strategy="auto",  # selects "cuaev" if CudaAEV extensions are available, pyaev if not
            neighborlist="cell_list",  # del if too slow #could try use_cuaev_interface=True above
        )

        # Mean and Standard Deviation values for normalizing the output
        self.ens_means = ens_means
        self.ens_stdevs = ens_stdevs

    @staticmethod
    def _load_models(model_paths):
        # Create a dictionary to store all models by atom type
        models = {}
        for atom_type, paths in model_paths.items():
            modules = []
            for path in paths:
                state = torch.load(path, map_location="cpu", weights_only=True)

                emb_dim = state["layer0.weight"].shape[
                    1
                ]  # set embedding dimension to first layer
                m = NMR(embedding=emb_dim)
                m.load_state_dict(state)
                m.eval()
                modules.append(m)

            models[atom_type] = torch.nn.ModuleList(modules)
        return torch.nn.ModuleDict(models)

    def forward(self, entry, batch_size=100):
        """
        Predicts the chemical shift for the given entry and batch size.

        Args:
            entry: Entry object containing simulation data
            batch_size: size of mini-batch for memory efficiency
        Returns:
            df_all: DataFrame of predicted chemical shifts for all atom types
        """
        assert (
            entry.species.dim() == 1
        ), "Species should be only for a single frame even it's a trajectory."

        num_frames = entry.coordinates.shape[0]
        # compute AEVs and run models for all batches
        cs_all_batches = self._compute_aevs_and_run_models(
            entry, batch_size, num_frames
        )

        # Combine results from all batches and all atom types into a single DataFrame
        df_all = self._combine_all_frames(cs_all_batches, num_frames, entry)
        return df_all

    def _compute_aevs_and_run_models(self, entry, batch_size, num_frames):
        cs_all_batches = []
        # We need to do mini-batch processing because of memory limitation
        num_batches = math.ceil(num_frames / batch_size)

        for i in range(num_batches):
            # Compute AEVs for the current batch
            aevs_batch = self.compute_batch_aev(entry, i, batch_size)

            # Run the models to predict the chemical shifts
            chemical_shifts = self.run_models(aevs_batch, entry.res_idx, entry.indices)

            cs_all_batches.append(chemical_shifts)

        return cs_all_batches

    def _combine_all_frames(self, cs_all_batches, num_frames, entry):
        # cs_all_batches is a list of dictionary, each dictionary is for one atom type
        # we need to merge each atom type's dictionary into a single dictionary that have all frames
        # then we can convert it to a dataframe
        df_all = []

        for atype in interested_atypes:
            cs_all_frames_atype = torch.cat(
                [batch[atype] for batch in cs_all_batches], dim=0
            )  # [num_frames, num_atoms]
            cs_std_all_frames_atype = torch.cat(
                [batch[atype + "_std"] for batch in cs_all_batches], dim=0
            )  # [num_frames, num_atoms]
            res_idx_atype = entry.res_idx.index_select(0, entry.indices[atype])
            seq_id_atype = torch.tensor(entry.seq_id).index_select(
                0, entry.indices[atype]
            )

            # Adjust for a single frame
            if num_frames == 1:
                cs_all_frames_atype = cs_all_frames_atype.flatten()
                cs_std_all_frames_atype = cs_std_all_frames_atype.flatten()
            else:
                cs_all_frames_atype = cs_all_frames_atype.transpose(
                    0, 1
                )  # [num_atoms, num_frames]
                cs_std_all_frames_atype = cs_std_all_frames_atype.transpose(0, 1)

            # Create DataFrame for current atom type and append to df_all
            data = {
                "ATOM_TYPE": [atype] * len(res_idx_atype),
                "SEQ_ID": seq_id_atype.flatten().tolist(),
                "RES_TYPE": [
                    entry.id_to_resname[r.item()] for r in res_idx_atype.flatten()
                ],
                "CHEMICAL_SHIFT": cs_all_frames_atype.tolist(),
                "CHEMICAL_SHIFT_STD": cs_std_all_frames_atype.tolist(),
            }
            df_all.append(pd.DataFrame(data))

        return pd.concat(df_all, ignore_index=True)

    def compute_batch_aev(self, entry, i, batch_size):
        """
        Calculate a batch of AEVs, cuaev could only process single molecule at a time
        """
        # Define batch boundaries
        start = i * batch_size
        num_frames = entry.coordinates.shape[0]
        end = min((i + 1) * batch_size, num_frames)

        # Create batch of AEVs
        aevs_all = []
        for j in range(start, end):
            aevs = self.aev_computer(
                entry.species.unsqueeze(0), entry.coordinates[j].unsqueeze(0)
            )
            aevs_all.append(aevs)
        aevs_all = torch.cat(aevs_all, 0)
        return aevs_all

    def run_models(self, aevs, res_idx, indices):
        cs_all = {}
        num_frames = aevs.shape[0]

        for atype in interested_atypes:
            # print(f"Predicting {atype} chemical shifts")
            # todo register indices as buffer
            aevs_atype = aevs.index_select(1, indices[atype])
            res_idx_atype = res_idx.index_select(0, indices[atype])
            res_idx_atype_expanded = res_idx_atype.unsqueeze(-1).expand(
                num_frames, -1, -1
            )  # [num_frames, num_atoms, 1]

            inputs = torch.cat(
                [aevs_atype, res_idx_atype_expanded], -1
            )  # [num_frames, num_atoms, num_features]
            inputs = inputs.flatten(0, 1)  # [num_frames * num_atoms, num_features]
            models_outputs = []
            for model in self.models[atype]:
                model = model
                model.eval()
                with torch.no_grad():
                    outputs = model(inputs)  # [num_frames * num_atoms]
                    models_outputs.append(outputs)

            models_avg = torch.stack(models_outputs).mean(0)  # [num_frames * num_atoms]
            models_std = self.denorm_chemical_shifts(
                torch.stack(models_outputs), atype, res_idx_atype_expanded.flatten()
            ).std(
                0
            )  # [num_frames * num_atoms]

            denormed_cs = self.denorm_chemical_shifts(
                models_avg, atype, res_idx_atype_expanded.flatten()
            )
            denormed_cs = denormed_cs.view(num_frames, -1)  # [num_frames, num_atoms]
            models_std = models_std.view(num_frames, -1)  # [num_frames, num_atoms]
            cs_all[atype] = denormed_cs
            cs_all[atype + "_std"] = models_std
        return cs_all

    def denorm_chemical_shifts(self, chemical_shift, atype, res_idx_atype):
        ens_stdevs_atype = self.ens_stdevs[atype].to(chemical_shift.device)
        ens_means_atype = self.ens_means[atype].to(chemical_shift.device)
        denormed_cs = (
            chemical_shift * ens_stdevs_atype[res_idx_atype]
            + ens_means_atype[res_idx_atype]
        )
        return denormed_cs


def load_and_validate_file(
    filepath: str,
    topology: Optional[str] = None,
    PDB: bool = False,
    TRAJECTORY: bool = False,
):
    """
    Loads and validates the file at the given file path.

    Args:
        filepath: path to file
        topology: path to topology file (optional)
        PDB: boolean, True if the file is a PDB file
        TRAJECTORY: boolean, True if the file is a trajectory file
    Returns:
        entry: Entry object with loaded data
    """
    if PDB:
        parser = PDBParser(PERMISSIVE=1)
        entryid = filepath.split("_")[0].strip()
        structure = parser.get_structure(entryid, filepath)

        assert len(structure) == 1
        entry = EntryPDB(structure[0])

    elif TRAJECTORY:
        if topology:
            traj = md.load(filepath, top=topology)
        else:
            traj = md.load(filepath)
        entry = EntryMdtraj(traj)

    if entry.unsupported:
        raise ValueError(
            f"PDB file has unsupported species at {entry.unsupported_atype}"
        )

    return entry


def parse_atypes(value):
    # Split the comma-separated string into a list of atom types
    atypes = value.split(",")
    # Return the atom types as a list of lists
    return atypes


def write_pdbCS(input_pdb_path, df, output_pdbcs_path):
    """
    Writes a new PDB file with chemical shifts replacing the B-factor column.
    Non-predicted atoms will have 'NA' instead of the B-factor.
    PDB files use fixed-width columns, so we limit decimal precision to avoid misalignment
    So, 13C and 15N shifts use only 1 decimal places, while 1H use 2 decimal places
    """
    # Group all chemical shifts per (SEQ_ID, ATOM_TYPE)
    shift_lookup = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )  # shift_lookup[seq_id][atype] = list of shifts

    for _, row in df.iterrows():
        seq_id = row["SEQ_ID"]
        atype = row["ATOM_TYPE"].strip()
        shift_lookup[seq_id][atype].append(row["CHEMICAL_SHIFT"])

    # Track how many of each ATOM_TYPE we've assigned per residue
    assigned_counts = collections.defaultdict(lambda: collections.defaultdict(int))

    with open(input_pdb_path, "r") as f_in, open(output_pdbcs_path, "w") as f_out:
        for line in f_in:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                res_seq = int(line[22:26].strip())

                # Special GLY handling: treat HA2/HA3 like HA
                # Normalize all H variants like 1HA/2HA/HA2/HA3 to HA to match model output
                #                key = (res_seq, normalized_atom_name)
                # Normalize: 1HA, 2HA → HA
                norm_atom_name = re.sub(r"\d", "", atom_name)

                i = assigned_counts[res_seq][norm_atom_name]

                if (
                    res_seq in shift_lookup
                    and norm_atom_name in shift_lookup[res_seq]
                    and i < len(shift_lookup[res_seq][norm_atom_name])
                ):
                    cs_value = shift_lookup[res_seq][norm_atom_name][i]
                    assigned_counts[res_seq][norm_atom_name] += 1

                    # Specify precision based on atom type (1 decimal for N, CA, CB, C; 2 decimals for H, HA)
                    # If the atom type is not in the lookup, set cs_str to "  NA  "

                    if norm_atom_name in ["N", "CA", "CB", "C"]:
                        cs_str = f"{cs_value:6.1f}"
                    elif norm_atom_name in ["H", "HA"]:
                        cs_str = f"{cs_value:6.2f}"
                    else:
                        cs_str = f"{cs_value:6.1f}"
                else:
                    cs_str = "  NA  "

                new_line = line[:60] + cs_str + line[66:]
                f_out.write(new_line)

            elif line.startswith("ANISOU"):
                continue

            else:
                f_out.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Path to input file(s), could be a PDB or netcdf trajectory file",
    )
    parser.add_argument("-t", "--topology", default=None)
    parser.add_argument(
        "-b", "--batch_size", default=10, type=int
    )  # you could change the batch size to fit your GPU memory
    parser.add_argument(
        "-atype",
        "--interested_atypes",
        default=["H", "CA", "CB", "C", "N", "HA"],
        type=parse_atypes,
        help="Interested atom types, the atypes should be separated by comma, for example H,HA",
    )
    parser.add_argument(
        "-o",
        "--output",
        choices=["csv", "parquet", "pdbcs", "all"],
        default="all",
        help="Output file format: csv, parquet, pdbcs, or all (default)",
    )
    args = parser.parse_args()
    interested_atypes = args.interested_atypes

    # Call 5 models
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_paths = {}
    for atype in interested_atypes:
        model_paths[atype] = [
            os.path.join(script_dir, "ens_models", f"ens_model_{i+1}_{atype}.pt")
            for i in range(NUM_MODELS)
        ]

    # Can accept single or multiple files
    for input_file in args.input_files:
        # Get the file extension
        _, file_extension = os.path.splitext(input_file)

        # Determine file type and load accordingly
        if file_extension.lower().startswith(".pdb"):
            entry = load_and_validate_file(input_file, PDB=True)
        elif file_extension.lower() == ".nc":
            if args.topology is None:
                raise ValueError("Topology file is required for trajectory files")
            entry = load_and_validate_file(
                input_file, topology=args.topology, TRAJECTORY=True
            )

        # Run LEGOLAS
        entry = entry.to(device)
        model = ChemicalShiftPredictor(model_paths).to(device)
        df = model(entry, args.batch_size)
        df = df.sort_values(by=["SEQ_ID", "ATOM_TYPE"]).reset_index(drop=True)
        print(df)

        stem = Path(input_file).stem

        # Save to CSV, Parquet, PDB
        if args.output in ["csv", "all"]:
            csv_file = stem + "_cs.csv"
            df.to_csv(csv_file, index=False)
            print(f"Saved to {csv_file}")

        if args.output in ["parquet", "all"]:
            parquet_file = stem + "_cs.parquet"
            df.to_parquet(parquet_file, index=False)
            print(f"Saved to {parquet_file}")

        if args.output in ["pdbcs", "all"] and file_extension.lower().startswith(
            ".pdb"
        ):
            pdbcs_file = stem + "_cs.pdb"
            write_pdbCS(input_file, df, pdbcs_file)
            print(f"Saved to {pdbcs_file}")

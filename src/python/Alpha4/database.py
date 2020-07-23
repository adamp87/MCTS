"""
Implementation of a database using the HDF5 library

Author: AdamP 2020-2020
"""

import h5py
import numpy as np


class Database:
    """
    Implementation of database to store input/output tensors for DNN training
    Stores the input state and output policy and value tensors.
    """
    def __init__(self, log, datafilepath_hdf, dims_state, dims_policy):
        self.log = log
        if not h5py.is_hdf5(datafilepath_hdf):
            Database._create_database(datafilepath_hdf, dims_state, dims_policy)
            self.log.info("Database have been created at {0}".format(datafilepath_hdf))
        self.datafile = h5py.File(datafilepath_hdf, "a")

    @staticmethod
    def _create_database(datafilepath_hdf, dims_state, dims_policy):
        datafile = h5py.File(datafilepath_hdf, "w")
        datafile.create_dataset("state", dtype=np.float32,
                                shape=(0, dims_state[0], dims_state[1], dims_state[2]),
                                maxshape=(None, dims_state[0], dims_state[1], dims_state[2]),
                                chunks=(42, dims_state[0], dims_state[1], dims_state[2]),
                                compression="gzip", compression_opts=1)
        datafile.create_dataset("policy", dtype=np.float32,
                                shape=(0, dims_policy[0], dims_policy[1], dims_policy[2]),
                                maxshape=(None, dims_policy[0], dims_policy[1], dims_policy[2]),
                                chunks=(42, dims_policy[0], dims_policy[1], dims_policy[2]),
                                compression="gzip", compression_opts=1)
        datafile.create_dataset("value", dtype=np.float32,
                                shape=(0, 1),
                                maxshape=(None, 1),
                                chunks=(1024 * 1024, 1),
                                compression="gzip", compression_opts=1)
        datafile.create_dataset("game_idx", dtype=np.uint32,
                                shape=(0, 1),
                                maxshape=(None, 1),
                                chunks=(1024 * 1024, 1),
                                compression="gzip", compression_opts=1)
        datafile.close()

    def store(self, game_idx, data_state, data_policy, data_result):
        """Stores the simulated data of one self-play game"""
        if len(data_state) != len(data_policy) != len(data_result):
            self.log.error("Error with result size")
            exit(-1)

        dset = self.datafile["state"]
        dset_idx = dset.shape[0]
        dset.resize(dset_idx+len(data_state), axis=0)
        dset[dset_idx:, :, :, :] = data_state

        dset = self.datafile["policy"]
        dset_idx = dset.shape[0]
        dset.resize(dset_idx+len(data_state), axis=0)
        dset[dset_idx:, :, :, :] = data_policy

        dset = self.datafile["value"]
        dset_idx = dset.shape[0]
        dset.resize(dset_idx+len(data_state), axis=0)
        dset[dset_idx:, 0] = data_result

        data_game_idx = np.zeros(len(data_state), dtype=np.uint32) + game_idx
        dset = self.datafile["game_idx"]
        dset_idx = dset.shape[0]
        dset.resize(dset_idx+len(data_state), axis=0)
        dset[dset_idx:, 0] = data_game_idx

        self.datafile.flush()

    def load(self, count):
        """
        Returns maximum 'count' number of samples from self-play games.
        The function selects random samples from the database.
        If database has less samples then 'count', all samples are returned.

        :return: state: Input state tensors
        :return: policy: Output policy tensor
        :return: value: Output value array
        """
        n_states = self.get_state_count()
        idx = np.random.choice(np.arange(0, n_states, 1), np.min((count, n_states)), replace=False)
        idx = np.sort(idx)  # hdf5 requires sorted index
        state = np.array(self.datafile["state"][idx, :, :, :])
        policy = np.array(self.datafile["policy"][idx, :, :, :])
        value = np.array(self.datafile["value"][idx, :])
        return state, policy, value

    def get_game_count(self):
        """
        Get the number of executed self-play games.
        note: Sequential storage in database is not guaranteed.
        """
        dset = self.datafile["game_idx"]
        count = dset.shape[0]
        if count == 0:
            return 0
        return np.max(dset)

    def get_state_count(self):
        """Get number of stored states (training samples)"""
        return self.datafile["state"].shape[0]

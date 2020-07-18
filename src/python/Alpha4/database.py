import h5py
import numpy as np


class Database:
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

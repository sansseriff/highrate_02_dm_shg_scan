# d is maximum middle
# r is minimum middle

from dataclasses import dataclass
import numpy as np
from enum import Enum
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import QuantumTomography as qKLib
from qutip import *


class Bin(Enum):
    EARLY = 0
    MIDDLE = 1
    LATE = 2


class Phase(Enum):
    DD = 0
    DR = 1
    RD = 2
    RR = 3


# @dataclass
# class BinBounds:
#     start: int | float
#     end: int | float


# @dataclass
# class BinRanges:
#     early: BinBounds
#     middle: BinBounds
#     late: BinBounds


@dataclass
class TomoCounts:
    def __init__(self):
        self.projections = [
            "ee",
            "el",
            "ed",
            "er",
            "le",
            "ll",
            "ld",
            "lr",
            "de",
            "dl",
            "dd",
            "dr",
            "re",
            "rl",
            "rd",
            "rr",
        ]

        # for the phase setting |dd>, which projections should be used?
        self.enabled_elements = {
            Phase.DD: [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            Phase.DR: [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
            Phase.RD: [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            Phase.RR: [1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1],
        }

        # the list of 16 3-bin projections. The leftmost column on the figure
        # showing all the terms that go into calculating the density matix.
        self.time_projections = [
            (Bin.EARLY, Bin.EARLY),  # |ee>
            (Bin.EARLY, Bin.LATE),  # |el>
            (Bin.EARLY, Bin.MIDDLE),  # |ed>
            (Bin.EARLY, Bin.MIDDLE),  # |er>
            (Bin.LATE, Bin.EARLY),  # |le>
            (Bin.LATE, Bin.LATE),  # |ll>
            (Bin.LATE, Bin.MIDDLE),  # |ld>
            (Bin.LATE, Bin.MIDDLE),  # |lr>
            (Bin.MIDDLE, Bin.EARLY),  # |de>
            (Bin.MIDDLE, Bin.LATE),  # |dl>
            (Bin.MIDDLE, Bin.MIDDLE),  # |dd>
            (Bin.MIDDLE, Bin.MIDDLE),  # |dr>
            (Bin.MIDDLE, Bin.EARLY),  # |re>
            (Bin.MIDDLE, Bin.LATE),  # |rl>
            (Bin.MIDDLE, Bin.MIDDLE),  # |rd>
            (Bin.MIDDLE, Bin.MIDDLE),  # |rr>
        ]

        self.ee: np.ndarray = np.zeros(4).astype("float64")
        self.el: np.ndarray = np.zeros(4).astype("float64")
        self.ed: np.ndarray = np.zeros(4).astype("float64")
        self.er: np.ndarray = np.zeros(4).astype("float64")
        self.le: np.ndarray = np.zeros(4).astype("float64")
        self.ll: np.ndarray = np.zeros(4).astype("float64")
        self.ld: np.ndarray = np.zeros(4).astype("float64")
        self.lr: np.ndarray = np.zeros(4).astype("float64")
        self.de: np.ndarray = np.zeros(4).astype("float64")
        self.dl: np.ndarray = np.zeros(4).astype("float64")
        self.dd: np.ndarray = np.zeros(4).astype("float64")
        self.dr: np.ndarray = np.zeros(4).astype("float64")
        self.re: np.ndarray = np.zeros(4).astype("float64")
        self.rl: np.ndarray = np.zeros(4).astype("float64")
        self.rd: np.ndarray = np.zeros(4).astype("float64")
        self.rr: np.ndarray = np.zeros(4).astype("float64")

        # self.rate_drc_coinc = 0

    def fill(
        self,
        phase: Phase,
        coinc_list_1: np.ndarray,
        coinc_list_2: np.ndarray,
        elapsed_time: float,
        bin_ranges: list[list[int]] = [[0, 75], [85, 155], [165, 240]],
    ):
        """fill a column of the array as shown in figure XX

        Args:
            phase (Bin): which phase setting/column to fill. \
                0 = |dd>, 1 = |dr>, etc.
            coinc_list_1 (list): list of time tags in histogram space \
                that share a coincidence with coinc_list_2
            coinc_list_2 (list): list of time tags in histogram space \
                that share a coincidence with coinc_list_1
            elapsed_time (float): time for the integration

        """
        enabled = self.enabled_elements[phase]

        for i, projection in enumerate(self.projections):
            # projection is something like 'dr' or 'ed'
            Bin1, Bin2 = self.time_projections[i]
            array = self.get_projection(projection)
            if enabled[i]:
                number, arr1, arr2 = dm_element(
                    coinc_list_1, coinc_list_2, Bin1, Bin2, bin_ranges=bin_ranges
                )
                rate = number / elapsed_time
                array[phase.value] = round(rate, 4)

    def kwiat_tomo_output(self):
        struct = np.array(
            [
                [1, 0, 0, np.nan, 1, 0, 1, 0],
                [1, 0, 0, np.nan, 1, 0, 0, 1],
                [1, 0, 0, np.nan, 1, 0, 0.7071, 0.7071],
                [1, 0, 0, np.nan, 1, 0, 0.7071, 0.7071j],
                [1, 0, 0, np.nan, 0, 1, 1, 0],
                [1, 0, 0, np.nan, 0, 1, 0, 1],
                [1, 0, 0, np.nan, 0, 1, 0.7071, 0.7071],
                [1, 0, 0, np.nan, 0, 1, 0.7071, 0.7071j],
                [1, 0, 0, np.nan, 0.7071, 0.7071, 1, 0],
                [1, 0, 0, np.nan, 0.7071, 0.7071, 0, 1],
                [1, 0, 0, np.nan, 0.7071, 0.7071, 0.7071, 0.7071],
                [1, 0, 0, np.nan, 0.7071, 0.7071, 0.7071, 0.7071j],
                [1, 0, 0, np.nan, 0.7071, 0.7071j, 1, 0],
                [1, 0, 0, np.nan, 0.7071, 0.7071j, 0, 1],
                [1, 0, 0, np.nan, 0.7071, 0.7071j, 0.7071, 0.7071],
                [1, 0, 0, np.nan, 0.7071, 0.7071j, 0.7071, 0.7071j],
            ]
        )

        for i, row in enumerate(struct):
            row[3] = self.get_row_by_idx(i).sum()

        return struct

    def get_projection(self, projection: str) -> np.ndarray:
        return self.__dict__[projection]

    def __str__(self):
        st = "phase\t\t|dd>\t\t|dr>\t\t|rd>\t\t|rr>\n"
        for item in self.projections:
            array = self.__dict__[item]
            array_st = create_matrix_row(array)
            st += f"{item}: {array_st}\n"
        return st

    def __call__(self):
        st = "phase\t\t|dd>\t\t|dr>\t\t|rd>\t\t|rr>\n"
        for item in self.projections:
            array = self.__dict__[item]
            array_st = create_matrix_row(array)
            st += f"{item}: {array_st}\n"
        return st

    def get_row_by_idx(self, idx: int) -> np.ndarray:
        return self.__dict__[self.projections[idx]]

    def calculate_density_matrix(self):
        self.t = qKLib.Tomography()
        self.t.importConf("./conf.txt")
        tomo_input = self.kwiat_tomo_output()
        intensity = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.rho, self.intens, self.fval = self.t.state_tomography(
            tomo_input, intensity
        )
        # convert to qutib object
        self.rho = Qobj(self.rho, dims=[[2, 2], [2, 2]])

    def calculate_car_mu(self):
        # a mu value derived from the CAR (coincidence to accidental ratio)
        return 1 / (np.average(0.5 * (self.ee / self.el + self.ll / self.le)))

    def calculate_log_negativity(self, rho):
        # from https://journals.aps.org/pra/pdf/10.1103/PhysRevA.65.032314
        rho_partial_transpose = partial_transpose(rho, [1, 0], method="dense")
        log_negativity = np.log(rho_partial_transpose.norm()) / np.log(2)
        self.log_negativity = log_negativity
        return log_negativity

    def calculate_coherent_information(self, rho):
        # from https://royalsocietypublishing.org/doi/abs/10.1098/rspa.2004.1372
        self.Iab = entropy_vn(rho.ptrace(0), base=2) - entropy_vn(rho, base=2)
        self.Iba = entropy_vn(rho.ptrace(1), base=2) - entropy_vn(rho, base=2)
        return max(self.Iab, self.Iba)

    def calculate_fidelity(self, dcr_subtraction = None):
        if dcr_subtraction is not None:
            return (1 - ((self.rr[3] - dcr_subtraction) / self.dd[0])) * 100
        else:
            return (1 - (self.rr[3] / self.dd[0])) * 100

    def calculate_visibility(self, dcr_subtraction = None):

        if dcr_subtraction is not None:
            return ((self.dd[0] - (self.rr[3] - dcr_subtraction)) / (self.dd[0] + self.rr[3] - dcr_subtraction)) * 100
        else:
            return ((self.dd[0] - self.rr[3]) / (self.dd[0] + self.rr[3])) * 100

    def calculate_qber(self):
        # from https://pubs.aip.org/aip/app/article/7/1/016106/2835124/Quantum-communication-with-time-bin-entanglement
        return (1 - self.calculate_visibility()/100) / 2


def create_matrix_row(array):
    array_st = "\t\t"
    for elem in array:
        elem_str = str(elem)
        tab_number = abs(2 - len(elem_str) // 8)
        tab_str = "\t" * tab_number
        array_st += f"{elem}{tab_str}"
    return array_st


def print_matrix(matrix):
    minimum_length = 0
    for row in matrix:
        for item in row:
            if len(str(item)) > minimum_length:
                minimum_length = len(str(item))
    maximum_tab_number = int(minimum_length // 8) + 1

    matrix_st = ""
    for row in matrix:
        row_st = ""
        for elem in row:
            elem_str = str(elem)
            tab_number = abs(maximum_tab_number - (len(elem_str) // 8))
            tab_str = "\t" * tab_number
            row_st += f"{elem}{tab_str}"
        matrix_st += row_st + "\n"

    return matrix_st


def dm_element(
    coincidences_array_1: np.ndarray,
    coincidences_array_2: np.ndarray,
    Bin1: Bin,
    Bin2: Bin,
    bin_ranges: list[list[int]] = [[0, 80], [90, 150], [160, 240]],
):
    # given two input arrays containing coincidences, output the number of counts in the selected bins

    coincidences_array_1 = np.array(coincidences_array_1)
    coincidences_array_2 = np.array(coincidences_array_2)
    mask_1 = (coincidences_array_1 >= bin_ranges[Bin1.value][0]) & (
        coincidences_array_1 <= bin_ranges[Bin1.value][1]
    )

    mask_2 = (coincidences_array_2 >= bin_ranges[Bin2.value][0]) & (
        coincidences_array_2 <= bin_ranges[Bin2.value][1]
    )
    filtered_mask = mask_1 & mask_2
    return (
        np.count_nonzero(coincidences_array_1[filtered_mask]),
        coincidences_array_1[filtered_mask],
        coincidences_array_2[filtered_mask],
    )


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


if __name__ == "__main__":
    matrix = np.random.rand(5, 5)
    # print(matrix)

    print(print_matrix(matrix))

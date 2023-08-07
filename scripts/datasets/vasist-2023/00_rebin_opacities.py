"""
Re-bin the opacity files to the resolution to the target resolution (R=400).

This implementation is based on the code from Vasist et al. (2023):
https://gitlab.uliege.be/francois.rozet/sbi-ear

See also the petitRADTRANS documentation:
https://petitradtrans.readthedocs.io/en/latest/content/notebooks/Rebinning_opacities.html
"""

import os

from molmass import Formula
from petitRADTRANS import Radtrans

if __name__ == "__main__":
    species = [
        "H2O_HITEMP",
        "CO_all_iso_HITEMP",
        "CH4",
        "NH3",
        "CO2",
        "H2S",
        "VO",
        "TiO_all_Exomol",
        "PH3",
        "Na_allard",
        "K_allard",
    ]

    masses = {
        s: Formula(s).isotope.massnumber
        for s in map(lambda s: s.split("_")[0], species)
    }

    path = os.path.join(
        os.environ["pRT_input_data_path"], "opacities/lines/corr_k"
    )

    atmosphere = Radtrans(line_species=species, wlen_bords_micron=[0.1, 251.0])
    atmosphere.write_out_rebin(400, path=path, species=species, masses=masses)

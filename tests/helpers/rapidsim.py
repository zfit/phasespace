#!/usr/bin/env python3
# =============================================================================
# @file   rapidsim.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   07.03.2019
# =============================================================================
"""Utils to crossheck against RapidSim."""

import os

import numpy as np
import uproot

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FONLL_FILE = os.path.join(BASE_PATH, "data", "fonll", "LHC{}{}.root")


def get_fonll_histos(energy, quark):
    with uproot.open(FONLL_FILE.format(quark, int(energy))) as histo_file:
        return histo_file["pT"], histo_file["eta"]


def generate_fonll(mass, beam_energy, quark, n_events):
    def analyze_histo(histo):
        x_axis = histo.axis(0)
        x_bins = x_axis.edges()
        bin_width = x_axis.width
        bin_centers = x_bins[:-1] + bin_width / 2
        normalized_values = histo.values() / np.sum(histo.values())
        return bin_centers, normalized_values

    pt_histo, eta_histo = get_fonll_histos(beam_energy, quark)
    pt_bin_centers, pt_normalized_values = analyze_histo(pt_histo)
    eta_bin_centers, eta_normalized_values = analyze_histo(eta_histo)
    pt_rand = np.random.choice(
        pt_bin_centers,
        size=n_events,
        p=pt_normalized_values,
    )
    pt_rand = 1_000 * np.abs(pt_rand)
    eta_rand = np.random.choice(
        eta_bin_centers,
        size=n_events,
        p=eta_normalized_values,
    )
    phi_rand = np.random.uniform(0, 2 * np.pi, size=n_events)
    px = pt_rand * np.cos(phi_rand)
    py = pt_rand * np.sin(phi_rand)
    pz = pt_rand * np.sinh(eta_rand)
    e = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    return np.stack([px, py, pz, e])


def load_generated_histos(file_name, particles):
    with uproot.open(file_name) as rapidsim_file:
        return {
            particle: [
                rapidsim_file.get(f"{particle}_{coord}_TRUE").array(library="np")
                for coord in ("PX", "PY", "PZ", "E")
            ]
            for particle in particles
        }


def get_tree(file_name, top_particle, particles):
    """Load a RapidSim tree."""
    with uproot.open(file_name) as rapidsim_file:
        tree = rapidsim_file["DecayTree"]
        return {
            particle: np.stack(
                [
                    1000.0 * tree[f"{particle}_{coord}_TRUE"].array(library="np")
                    for coord in ("PX", "PY", "PZ", "E")
                ]
            )
            for particle in particles
        }


def get_tree_in_b_rest_frame(file_name, top_particle, particles):
    def lorentz_boost(part_to_boost, boost):
        """
        Perform Lorentz boost
            vector :     4-vector to be boosted
            boostvector: boost vector. Can be either 3-vector or 4-vector (only spatial components
            are used)
        """
        boost_vec = -boost[:3, :] / boost[3, :]
        b2 = np.sum(boost_vec * boost_vec, axis=0)
        gamma = 1.0 / np.sqrt(1.0 - b2)
        gamma2 = (gamma - 1.0) / b2
        part_time = part_to_boost[3, :]
        part_space = part_to_boost[:3, :]
        bp = np.sum(part_space * boost_vec, axis=0)
        return np.concatenate(
            [
                part_space + (gamma2 * bp + gamma * part_time) * boost_vec,
                np.expand_dims(gamma * (part_time + bp), axis=0),
            ],
            axis=0,
        )

    part_dict = get_tree(file_name, top_particle, list(particles) + [top_particle])
    top_parts = part_dict.pop(top_particle)
    return {
        part_name: lorentz_boost(part, top_parts)
        for part_name, part in part_dict.items()
    }


# EOF

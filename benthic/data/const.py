r"""Information about our dataset"""

MONTHLY_INDICES = {
    "Jan": list(range(0, 31)),
    "Feb": list(range(31, 59)),
    "Mar": list(range(59, 90)),
    "Apr": list(range(90, 120)),
    "May": list(range(120, 151)),
    "Jun": list(range(151, 181)),
    "Jul": list(range(181, 212)),
    "Aug": list(range(212, 243)),
    "Sep": list(range(243, 273)),
    "Oct": list(range(273, 304)),
    "Nov": list(range(304, 334)),
    "Dec": list(range(334, 365)),
}
SEASONAL_INDICES = {
    "Winter": list(range(0, 90)),
    "Spring": list(range(90, 181)),
    "Summer": list(range(181, 273)),
    "Fall": list(range(273, 365)),
}

DATASET_INPUT_VARIABLES = [
    "bathymetry",
    "gravel",
    "mud_sand_ratio",
    "votemper",
    "vosaline",
    "dox",
    "sCSED",
    "botfluxPOC",
    "botstress",
    "fCSED",
    "par",
]


OUTPUT_PROPORTION_GROUPS = {
    "body_mass": [
        "body_mass_0.001",
        "body_mass_0.001_0.010",
        "body_mass_0.010_0.100",
        "body_mass_0.100_1.000",
        "body_mass_1.000_10.000",
        #"body_mass_10.000",
    ],
    "life_span": [
        "life_span_1",
        "life_span_1_3",
        "life_span_3_10",
        "life_span_10_20",
        "life_span_20",
    ],
    "sexuality": [
        "sexuality_gonochorism",
        "sexuality_homogamy",
        "sexuality_protandry",
    ],
    "offspring_development": [
        "offspring_development_internal",
        "offspring_development_lecithotrophic",
        "offspring_development_mixed_lecithotrophic",
        "offspring_development_mixed_planktotrophic",
        "offspring_development_planktotrophic",
    ],
    "mobility": [
        "mobility_immobile",
        "mobility_limited",
        "mobility_slow",
        "mobility_fast",
        "mobility_very_fast",
    ],
    "posegrp": [
        "posegrp1",
        "posegrp2",
        "posegrp3",
    ],
    "feeding_type": [
        "feeding_type_casc",
        "feeding_type_de",
        "feeding_type_hegr",
        "feeding_type_su",
    ],
    "substratum_depth_occupancy": [
        "substratum_depth_occupancy_0",
        "substratum_depth_occupancy_0_5",
        "substratum_depth_occupancy_5_15",
        "substratum_depth_occupancy_15_30",
        "substratum_depth_occupancy_30",
    ],
}
 
# Effect groups (Effect_mod): modalities sum to the station total abundance,
# which is the same for every group of a station. 
OUTPUT_EFFECT_GROUPS = {
    "endo_bioconstruction_type": [
        "endo_bioconstruction_type_none",
        "endo_bioconstruction_type_blind_ended",
        "endo_bioconstruction_type_open_ended",
    ],
    "endo_bioconstruction_depth": [
        "endo_bioconstruction_depth_none_surficial",
        "endo_bioconstruction_depth_0_5",
        "endo_bioconstruction_depth_5_15",
        "endo_bioconstruction_depth_15_30",
        "endo_bioconstruction_depth_30",
    ],
    "sediment_mixing_type": [
        "sediment_mixing_type_none",
        "sediment_mixing_type_diffusion",
        "sediment_mixing_type_upward_conveying",
        "sediment_mixing_type_downward_conveying",
        "sediment_mixing_type_regeneration",
    ],
    "ventilation_pumping": [
        "ventilation_pumping_null",
        "ventilation_pumping_low",
        "ventilation_pumping_high",
    ],
}
 
# Independent outputs (ind_ecol): each scaled to [0, 1], no sum constraint.
OUTPUT_INDEPENDENT_VARIABLES = [
    "de",
    "mi",
    "ir",
    "dep",
]
 
# Column order of the output matrix: proportion groups, effect groups, independent.
# The structured model relies on this order (each group is a contiguous slice).
DATASET_OUTPUT_VARIABLES = [
    *[v for group in OUTPUT_PROPORTION_GROUPS.values() for v in group],
    *[v for group in OUTPUT_EFFECT_GROUPS.values() for v in group],
    *OUTPUT_INDEPENDENT_VARIABLES,
]

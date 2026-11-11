# Data provenance and attribution

## Original dataset

Source: COVID Moonshot cleaned experimental data

Repository:
https://github.com/asapdiscovery/COVID_moonshot_data_clean

Pinned source commit:
bb7317c5f8c122de5190fbaf7fc9e92d1077841b

The data originates from the COVID Moonshot project and
contains experimental SARS-CoV-2 Mpro IC50 measurements.

## Source files

- cdd_noncovalent_dates_2023_10_18_filt.csv
- cdd_achiral_enantiopure_dates_2023_10_18_filt.csv

The second file is a subset of the first and is not
treated as an independent validation dataset.

## Derived data

The quantitative compound-level table was derived from
the noncovalent dataset.

Quantitative measurements are aggregated within compound
IDs using the median of log10(IC50 in micromolar).

Censored measurements are excluded from the regression
targets but their presence is recorded in the compound
table.

Molecular features, split definitions, and audit reports
are derived artifacts produced by this project's scripts.

## Licensing

The upstream repository is MIT-licensed.

Its original copyright and license notice is preserved in:

docs/third_party/COVID_moonshot_data_clean_LICENSE

This attribution does not imply endorsement by the
original data contributors.

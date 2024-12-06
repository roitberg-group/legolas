import torchani
from torchani.aev import StandardRadial, StandardAngular

radial_terms = StandardRadial.like_2x(cutoff=10.4, num_shifts=32, exact=False)
angular_terms = StandardAngular.like_2x(cutoff=7.0, num_shifts=16, exact=False)
aev_computer = torchani.AEVComputer(angular_terms=angular_terms,
                                    radial_terms=radial_terms,
                                    num_species=5)

breakpoint()

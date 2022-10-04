# relativebilbying

This code is a complement to the `bilby` package and enables one to perform relative binning in [`bilby`](https://git.ligo.org/tomasz.baka/bilby). We note that, for the moment, the code only implements relative binning without higher order modes.

Relative binning is a method to perform bayesian parameter inference for gravitational waves in a faster way. It has been developped in [Zackay et al](https://arxiv.org/pdf/1806.08792.pdf), [Dai et al](https://arxiv.org/pdf/1806.08793.pdf). Additionally, attempts have been made to develop relative binning for higher-order modes waveforms (see [Leslie et al](https://arxiv.org/pdf/2109.09872.pdf)). 

In this repo, we use the ideas from [Zackay et al](https://arxiv.org/pdf/1806.08792.pdf) and convert them into a bilby compatible format, making it easier to use.

If you use this code for your work, please give due credits by citating the methodology papers:

@article{Zackay:2018qdy,
    author = "Zackay, Barak and Dai, Liang and Venumadhav, Tejaswi",
    title = "{Relative Binning and Fast Likelihood Evaluation for Gravitational Wave Parameter Estimation}",
    eprint = "1806.08792",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    month = "6",
    year = "2018"
}

@article{Dai:2018dca,
    author = "Dai, Liang and Venumadhav, Tejaswi and Zackay, Barak",
    title = "{Parameter Estimation for GW170817 using Relative Binning}",
    eprint = "1806.08793",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "6",
    year = "2018"
}

(for higher order modes:
@article{Leslie:2021ssu,
    author = "Leslie, Nathaniel and Dai, Liang and Pratten, Geraint",
    title = "{Mode-by-mode relative binning: Fast likelihood estimation for gravitational waveforms with spin-orbit precession and multiple harmonics}",
    eprint = "2109.09872",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.1103/PhysRevD.104.123030",
    journal = "Phys. Rev. D",
    volume = "104",
    number = "12",
    pages = "123030",
    year = "2021"
}
)


And also citing this repository:
@misc{janquart:2022,
author = "Janquart, Justin and Puecher, Anna",
title="{RelativeBilbying: a package for relative binning with bilby}",
year={2022},
howpublished={\url{https://github.com/lemnis12/relativebilbying}}
}


### Installation
a) Using PyPI, you can just run  `pip install relativebilbying`
b) using this repo:
- clone the repository
- run `python setup.py install`


### Structure of the repo
- relativebilbying: folder with the likelihoods to perfrom relative binning in bilby
- example: folder with example runs for injection on how to use these scripts. 
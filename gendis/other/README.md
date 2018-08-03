# Other shapelet discovery algorithms

This directory contains implementations of other algorithms. The list of implemented algorithms is the following:
* Brute Force: A simple brute-force extraction [[PDF](http://alumni.cs.ucr.edu/~lexiangy/Shapelet/kdd2009shapelet.pdf)]
* Fast Shapelets: The optimizations proposed in the Logical Shapelets paper [[PDF](www.cs.ucr.edu/~neal/Mueen11Logical.pdf)]
* SAX Shapelets: Finds a suboptimal shapelet quickly (fastest) [[PDF](http://alumni.cs.ucr.edu/~rakthant/paper/13SDM_FastShapelets.pdf)]
* Particle Swarm Optimization: Another bio-inspired algorithm. [[PDF](http://www.ijmlc.org/vol5/521-C016.pdf)]

For an excellent implementation of Learning Timeseries Shapelets (LTS) ([[PDF](https://www.ismll.uni-hildesheim.de/pub/pdfs/grabocka2014e-kdd.pdf)]), check out [tslearn](https://tslearn.readthedocs.io/en/latest/).

Moreover, all algorithms try to optimize information gain, but other metrics are possible as well, the list of all metrics is:

* Information gain
* F-statistic
* Mood's median
* Kruskal-Wallis statistic
* _Bhattacharyya distance_
* _Class-Scatter matrix_

The ones annotated cursively can also be used to evaluate a set of shapelets, instead of a single shapelet.

**Please note that a lot of improvements are possible for these implementations. I do _NOT_ provide any support should any problems with one of these implementations arise.**

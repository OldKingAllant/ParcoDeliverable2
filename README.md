# Deliverable 2 for Parallel Computing at UniTn for year 24/25

# Objective

During the course, we studied MPI and how to use it to 
parallelize algorithms on distributed-memory systems.

As such, in this project I am trying to implement different
algorithms for matrix transposition and checking whether a
matrix is symmetric or not, by using MPI. After that, 
I want to compare the performance of these implementations
against the sequential naive implementation of those
two algorithms.

# Obtaining the code

````
mkdir -p claudio_vozza_d2 && cd claudio_vozza_d2
git clone https://github.com/OldKingAllant/ParcoDeliverable2.git
````

N.B.! The rest of the guide assumes that you put the 
'claudio_vozza_d2' folder in the root of your home
directory

# Rules for running 

The code expects:
- The number of elements to be a power of two
- The number of MPI processes to be a power of two
- The number of elements should be greater than the number of processes

# Compile and run locally

Disclaimer: 
````
This was tested locally on Windows
using Microsoft's MPI implementation, so
for that OS, you need to install the MPI SDK.
On linux, the normal installation of OpenMPI
will suffice
````

First and foremost, you need cmake 3.15+ and g++
version 4.5+, together with GNU Make. 

After obtaining the code as instructed above:
````
cd ParcoDeliverable2
cmake -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" .
make 
````

Now you can run locally (from claudio_vozza_d2/ParcoDeliverable2):
````
cd ParcoDeliverable2
mpirun -np <num_processes> ./ParcoDeliverable2 <num_elements> [result directory]
````

If the 'result directory' is omitted, the benchmarks will
be stored in the executable's directory

# Compile and run on the cluster

After accessing the cluster with your account, make
sure to have obtained the code and that the 
claudio_vozza_d2 exists in the root
of your home directory.

You can change the number of MPI processes
and number of elements by modifying the file at
claudio_vozza_d2/ParcoDeliverable2/ParcoDeliverable2/cluster_run.pbs
by adding/removing elements from the arrays
'all_n_elements' and 'all_n_procs' (while
respecting the rules stated in the second section)

From the root of your home dir:

````
cd claudio_vozza_d2/ParcoDeliverable2
mkdir -p run && cd run
qsub -v "out_dir=$(pwd)" ../ParcoDeliverable2/cluster_run.pbs
````

Wait at most 10 minutes before checking the results, 
which will appear as many text files in the same
directory from which you ran the pbs script. To see
if any errors occurred, cat the contents of
errors.txt (also bench.txt could contain
interesting things).

Still, if you want to run the test for exactly N processes
and M elements, you should go in the directory 
claudio_vozza_d2/ParcoDeliverable2/ParcoDeliverable2
and run the executable as you would do locally.
However, to do this, you should first run an interactive job:
````
qsub -I -q short_cpuQ -l select=1:ncpus=<max_cpus>:mpiprocs=<max_procs>
````
And run the executable from inside that job

# Compile flags explanation

Example command line used when compiling a single source file:
````
g++ -O3 -DNDEBUG   -Wall -msse4.1 -std=gnu++11 -o <object> -c <.cpp>
````

When linking:
````
g++  -O3 -DNDEBUG  -flto <list of object files> -o ParcoDeliverable2 -lmpi -lmpi_cxx
````

- Wall Shows all warnings (excluding pedantic)
- O3 Since we want max. optimizations when compiling (while
  producing stable and more or less machine-independent code)
- flto Used to perform link-time optimization (which should allow
  function inlining across multiple translation units)
- msse4.1 Enables using intrinsics for sse4.1 instructions
- lmpi and lmpi_cxx Tells the linker to use the MPI libraries
- std=c++11 Set CXX standard to 11

# Loaded modules

Necessary modules are automatically loaded by the PBS
script, but I will still write them out:
- cmake-3.15.4
- make-4.3
- openmpi-4.0.4

If you still want to load them manually, use
module load <name> for all of them

# How to interpret results

Example run:
````
mpirun -np 8 ./ParcoDeliverable2 4096
````

This will create a file called 'bench_8_4096.txt' 
inside the executable's directory.
So, the name format of the output file 
is bench_<procs>_<elems>.txt. 

The file will contain something like this:
````
270.885
135.471
44.6778
8.06175
97.1259
37.3783
11.7023
````

Each line corresponds to the time taken for
each algorithm to complete (more precisely, the average
time of 10 runs). 
The first 4 lines are used by the transposition
algs., while the remaining 3 are
the symmetry checks. 

To generate graphs, a python script is provided, which uses 
matplotlib (check if this dependency is installed beforehand).

Example run
````
python graphs.py 8 "1,2,4,8" "16,32,64,128,256,512,1024,2048,4096" ./bench_dir ./out_dir
````

In general, the structure of the arguments is:
````
python graphs.py <compare_procs> <list_procs> <list_elems> <input_dir> [out_dir]
````

- compare_procs: Number of processes to uses when comparint each algorithm to baseline
- list_procs: List of number of processes used for each benchmark
- list_elems: List of number of elements for each benchmark
- input_dir: Input directory where the 'bench_<procs>_<elems>.txt' reside
- out_dir: Optional output directory for the graphs (default is ./ when omitted)

In the output directory you will find 7 images which contain the graphs.

The two called 
- compare_symm_fixed_procs.png
- compare_transpose_fixed_procs.png

Represent the performance of the MPI algorithms wrt the baseline and the
other MPI implementations (for the given number of processes)

The remaining graphs compare the performance
scaling of each MPI algorithm when changing
the number of processes and the number of elements

# General code structure
- CMakeLists.txt: top level cmake script, not interesting
- graphs.py: Python script used to generate performance graphs

Files in ParcoDeliverable2:
- Bench.h: File with functions for benchmarking
- CMakeLists.txt: Real cmake script which configures compilation
- Defs.hpp: Global definitions
- Parallel.hpp/cpp: Contains MPI algorithms
- ParcoDeliverable2.hpp/cpp: Contains main function which orchestrates the execution
- Seq.hpp/cpp: Contains sequential algorithms
- Utils.hpp/cpp: Useful functions (like matrix init.)
- cluster_run.pbs: Script used to run the job on the cluster
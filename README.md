# Deliverable 2 for Parallel Computing at UniTn for year 24/25

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

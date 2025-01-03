import matplotlib as mp
import matplotlib.pyplot as plt
import math
import sys
import os

#1 : base transpose
#2 : column
#3 : block
#4 : root
#5 : distributed
#6 : base symm
#7 : block 
#8 : distributed
def process_file(file):
	names = ['base_transpose', 'block_transpose', 'root_transpose']
	names = names + ['distributed_transpose', 'base_symm', 'block_symm', 'distributed_symm']
	data = {}
	for name in names:
		time_taken = float(file.readline().rstrip())
		data[name] = time_taken
	return data

def gather_files(array_of_proc, array_of_elem, input_dir):
	data = []
	for num_proc in array_of_proc:
		for num_elem in array_of_elem:
			current_file = input_dir + '/bench_' + str(num_proc) + '_' + str(num_elem) + '.txt'
			with open(current_file) as input_file:
				data.append({'proc': num_proc, 'elem': num_elem, 'data': process_file(input_file)})
	return data

def generate_transpose(data, num_proc, out_dir, n_elements):
	all_elems = list(filter(lambda entry: entry['proc'] == num_proc, data))
	out_file = out_dir + '/compare_transpose_fixed_procs.png'
	print(f'Output file {out_file}')

	base_transpose = [data_per_elem['data']['base_transpose'] for data_per_elem in all_elems]
	block_transpose = [data_per_elem['data']['block_transpose'] for data_per_elem in all_elems]
	root_transpose = [data_per_elem['data']['root_transpose'] for data_per_elem in all_elems]
	distributed_transpose = [data_per_elem['data']['distributed_transpose'] for data_per_elem in all_elems]
	
	plt.clf()
	plt.title(f'Comparing MPI transposes to baseline (using {num_proc} processes)')
	plt.ylabel('Time (ms)')
	plt.xlabel('Matrix size (sqrt(N) elements)')

	plt.plot(n_elements, base_transpose, '^-b', label='Base transpose')
	plt.plot(n_elements, block_transpose, 's-r', label='Block transpose')
	plt.plot(n_elements, root_transpose, 'D-g', label='Root transpose')
	plt.plot(n_elements, distributed_transpose, 'o-y', label='Distributed transpose')
	
	plt.legend()
	#plt.show()
	plt.savefig(out_file)
	return

def generate_symm(data, num_proc, out_dir, n_elements):
	all_elems = list(filter(lambda entry: entry['proc'] == num_proc, data))
	out_file = out_dir + '/compare_symm_fixed_procs.png'
	print(f'Output file {out_file}')

	base_symm = [data_per_elem['data']['base_symm'] for data_per_elem in all_elems]
	block_symm = [data_per_elem['data']['block_symm'] for data_per_elem in all_elems]
	distributed_symm = [data_per_elem['data']['distributed_symm'] for data_per_elem in all_elems]
	
	plt.clf()
	plt.title(f'Comparing MPI symmetry to baseline (using {num_proc} processes)')
	plt.ylabel('Time (ms)')
	plt.xlabel('Matrix size (sqrt(N) elements)')

	plt.plot(n_elements, base_symm, '^-b', label='Base symm')
	plt.plot(n_elements, block_symm, 's-r', label='Block symm')
	plt.plot(n_elements, distributed_symm, 'D-g', label='Distributed symm')
	
	plt.legend()
	plt.savefig(out_file)
	return

def generate_mpi(data, out_dir, n_elements, n_procs, ty):
	out_file = out_dir + f'/compare_mpi_{ty}.png'
	print(f'Output file {out_file}')

	plt.clf()
	plt.title(f'Compare {ty} performance scaling based on number of processes')
	plt.ylabel('Time (ms)')
	plt.xlabel('Matrix size (sqrt(N) elements)')

	for curr_procs in n_procs:
		filtered_by_proc = list(filter(lambda entry: entry['proc'] == curr_procs, data))
		times = [entry['data'][ty] for entry in filtered_by_proc]
		plt.plot(n_elements, times, label=f'{curr_procs} processes')

	plt.legend()
	#plt.show()
	plt.savefig(out_file)
	return

def main():
	if len(sys.argv) < 5:
		print("Not enough arguments")
		return

	save_dir = "./"
	if len(sys.argv) == 6:
		save_dir = sys.argv[5]
		print(f'Save to: {save_dir}')
		if not os.path.isdir(save_dir):
			print('Creating save directory')
			os.makedirs(save_dir)

	
	num_proc = int(sys.argv[1])
	array_of_proc = [int(num) for num in sys.argv[2].split(',')]
	array_of_elem = [int(num) for num in sys.argv[3].split(',')]
	input_dir = sys.argv[4]

	print(f'Select processes: {num_proc}')
	print(f'Num processes: {array_of_proc}')
	print(f'Num elements: {array_of_elem}')
	print(f'Input directory: {input_dir}\n')

	data = gather_files(array_of_proc, array_of_elem, input_dir)

	#print(data)

	generate_transpose(data, num_proc, save_dir, array_of_elem)
	generate_symm(data, num_proc, save_dir, array_of_elem)

	generate_mpi(data, save_dir, array_of_elem, array_of_proc, 'block_transpose')
	generate_mpi(data, save_dir, array_of_elem, array_of_proc, 'root_transpose')
	generate_mpi(data, save_dir, array_of_elem, array_of_proc, 'distributed_transpose')
	generate_mpi(data, save_dir, array_of_elem, array_of_proc, 'block_symm')
	generate_mpi(data, save_dir, array_of_elem, array_of_proc, 'distributed_symm')

	return

#arg 1: select number of processes to compare against the baseline
#arg 2: array of number of processes, enclosed in double amps, like "1,2,4"
#arg 3: same form as 2, but for the number of elements
#arg 4: input directory where all files are found
#arg 5 (optional): directory where the script will save the graphs
if __name__ == '__main__':
	main()
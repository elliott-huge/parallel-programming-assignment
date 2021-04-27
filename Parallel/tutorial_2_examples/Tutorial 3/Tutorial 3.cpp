#include <iostream>
#include <vector>

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0;}
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		typedef int mytype;

		//Part 3 - memory allocation
		//host - input
		std::vector<mytype> A(555, 3);

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 256;

		size_t padding_size = A.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size-padding_size, 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		// B vector
		std::vector<mytype> B;

		// begin while?
		// while Bsize > workgroup size
		do while (B.size() > 1)
		{
			// set sizing
			size_t input_element_count = A.size();
			size_t input_size = A.size() * sizeof(mytype);
			size_t nr_groups = input_element_count / local_size;

			B.resize(nr_groups, 0);
			B.shrink_to_fit();

			size_t output_size = B.size() * sizeof(mytype);

			//set buffers
			cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
			cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
			queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

			//run kernels
			cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_assignment_step_1");
			kernel_1.setArg(0, buffer_A);
			kernel_1.setArg(1, buffer_B);
			kernel_1.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

			queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_element_count), cl::NDRange(local_size));

			//retrieve output
			queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

			//test
			A.resize(B.size());
			A.shrink_to_fit();
			A = B;

			// then assign array B's values to array A after resizing
			// resize array B (update nr_groups)
			// recalculate 
			// /loop
			
		}
		
		//host - output
		size_t output_size = B.size()*sizeof(mytype);//size in bytes

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

		//Part 4 - device operations

		//4.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

		//4.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_assignment_step_1");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

		//call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_element_count), cl::NDRange(local_size));

		
		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		//std::cout << "A = " << A << std::endl;
		std::cout << "B = " << B << std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}